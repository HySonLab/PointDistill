import unittest
import torch
import torch.nn as nn
import numpy as np
from gtda.homology import VietorisRipsPersistence
from pytorch3d.loss import chamfer_distance

try:
    from torch_cluster import knn

    KNN_AVAILABLE = True
except ImportError:
    KNN_AVAILABLE = False


def optimized_cdist(x, y=None, chunk_size=None):
    """
    Compute squared Euclidean distance matrix between x (n×d) and y (m×d) or x (n×d) if y is None.

    Args:
        x (torch.Tensor): Input tensor of shape (n, d).
        y (torch.Tensor, optional): Input tensor of shape (m, d). Defaults to x.
        chunk_size (int, optional): Number of rows of x to process at a time. Defaults to None (process all).

    Returns:
        torch.Tensor: Distance matrix of shape (n, m) where dist[i,j] = ||x[i,:] - y[j,:]||^2.
    """
    if y is None:
        y = x
    n, d = x.shape
    m = y.shape[0]

    # Compute norms: ||x[i,:]||^2 and ||y[j,:]||^2
    x_norm = torch.sum(x ** 2, dim=1, keepdim=True)  # Shape: (n, 1)
    y_norm = torch.sum(y ** 2, dim=1, keepdim=True).t()  # Shape: (1, m)

    if chunk_size is None or chunk_size >= n:
        # Compute -2 * x @ y^T
        xy = -2.0 * torch.mm(x, y.t())  # Shape: (n, m)
        # Compute distances: x_norm + y_norm - 2 * x @ y^T
        dist = x_norm + y_norm + xy
        # Clamp to avoid numerical errors
        dist = torch.clamp(dist, min=0.0)
        return dist
    else:
        # Process in chunks to reduce memory usage
        dist = torch.zeros(n, m, device=x.device, dtype=x.dtype)
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            x_chunk = x[i:end_i]  # Shape: (chunk_size, d)
            x_chunk_norm = x_norm[i:end_i]  # Shape: (chunk_size, 1)
            xy_chunk = -2.0 * torch.mm(x_chunk, y.t())  # Shape: (chunk_size, m)
            dist[i:end_i] = x_chunk_norm + y_norm + xy_chunk
        dist = torch.clamp(dist, min=0.0)
        return dist


class DifferentiableVietorisRipsPersistence:
    def __init__(self, homology_dimensions=[0, 1], max_edge_length=2.0,
                 max_edges=100000, max_cycles=100,
                 use_knn=False, knn_neighbors=20,
                 auto_config=False, distance_method="optimized_cdist"):

        self.homology_dimensions = homology_dimensions
        self.max_edge_length = max_edge_length
        self.max_edges = max_edges
        self.max_cycles = max_cycles
        self.use_knn = use_knn and KNN_AVAILABLE  # Fall back to non-kNN if torch-cluster is unavailable
        self.knn_neighbors = knn_neighbors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.use_knn and not KNN_AVAILABLE:
            print("Warning: torch-cluster not available, falling back to non-kNN method.")

        self.auto_config = auto_config

        # Accelerate matrix calculation
        self.distance_method = distance_method if distance_method in ["cdist", "optimized_cdist"] else "cdist"

    def auto_configure(self, points):
        """
        Automatically select max_edge_length and max_edges based on point cloud properties.
        Args:
            points: torch.Tensor of shape (n_points, n_features)
        Returns:
            max_edge_length: float
            max_edges: int
        """
        n_points = points.shape[0]
        sample_size = min(n_points, 5000)  # Sample for large clouds
        sample_indices = torch.randperm(n_points)[:sample_size]
        sample_points = points[sample_indices]

        # Estimate edge distances (use k-NN for efficiency if available)
        if self.use_knn and KNN_AVAILABLE:
            k = 5
            row, col = knn(sample_points, sample_points, k, batch_x=None, batch_y=None)
            mask = row < col
            edge_dists = torch.norm(sample_points[row[mask]] - sample_points[col[mask]], dim=-1)
        else:
            dist = torch.cdist(sample_points, sample_points, p=2)
            mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
            edge_dists = dist[mask]

        # Set max_edge_length as 3× the 90th percentile distance
        if edge_dists.numel() > 0:
            max_edge_length = torch.quantile(edge_dists, 0.9).item() * 3.0
        else:
            max_edge_length = 1.0  # Fallback for sparse clouds

        # Estimate total edges (approximate for k-NN or full matrix)
        if self.use_knn:
            total_edges = n_points * 20 / 2  # k=20, halved for deduplication
        else:
            total_edges = n_points * (n_points - 1) / 2  # Full matrix upper triangle
        # Set max_edges as 5% of total edges or 2× n_points, capped at 100,000
        max_edges = min(int(0.05 * total_edges), 2 * n_points, 100000)
        max_edges = max(max_edges, 1000)  # Ensure minimum for small clouds

        return max_edge_length, max_edges

    def _compute_distance_matrix(self, points):
        """
        Compute pairwise distances using k-NN or full distance matrix.
        Args:
            points: torch.Tensor of shape (n_points, n_features)
        Returns:
            edge_dists: torch.Tensor of shape (n_edges,), distances of valid edges
            edge_pairs: torch.Tensor of shape (n_edges, 2), pairs of point indices
        """
        n_points = points.shape[0]
        if n_points <= 1:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device, dtype=torch.long)

        # Auto-configure max_edge_length and max_edges if not set
        if self.auto_config:
            self.max_edge_length, self.max_edges = self.auto_configure(points)
            # print(f"--> Auto-configured: max_edge_length={self.max_edge_length:.4f}, max_edges={self.max_edges}")

        if self.use_knn:
            # k-NN version
            k = self.knn_neighbors  # Number of neighbors
            row, col = knn(points, points, k, batch_x=None, batch_y=None)
            mask = row < col
            edge_pairs = torch.stack([row[mask], col[mask]], dim=1)
            edge_dists = torch.norm(points[row[mask]] - points[col[mask]], dim=-1)
            mask = edge_dists <= self.max_edge_length
            return edge_dists[mask], edge_pairs[mask]
        else:
            # Full distance matrix version with dynamic chunking
            max_chunk_size = 10000  # Maximum chunk size for large point clouds
            chunk_size = min(n_points, max_chunk_size)
            edge_dists = []
            edge_pairs = []
            chunk_count = 0
            for i in range(0, n_points, chunk_size):
                points_i = points[i:i + chunk_size]

                if self.distance_method == "optimized_cdist":
                    dist = optimized_cdist(points_i, points, chunk_size=chunk_size)
                else:
                    dist = torch.cdist(points_i, points, p=2)  # Shape: (chunk_size, n_points)

                mask = dist <= self.max_edge_length
                if mask.any():
                    mask = mask & (torch.ones_like(dist, dtype=torch.bool).triu(diagonal=1)[:dist.shape[0], :])
                    chunk_dists = dist[mask]
                    row, col = torch.where(mask)
                    row = row + i
                    chunk_pairs = torch.stack([row, col], dim=1)
                    edge_dists.append(chunk_dists)
                    edge_pairs.append(chunk_pairs)
                chunk_count += 1
            # print(f"Processed {chunk_count} chunks")
            edge_dists = torch.cat(edge_dists) if edge_dists else torch.tensor([], device=self.device)
            edge_pairs = torch.cat(edge_pairs) if edge_pairs else torch.tensor([], device=self.device, dtype=torch.long)
            # self.max_edges = max_edges
            return edge_dists, edge_pairs

    def _approximate_dim0(self, edge_dists, edge_pairs):
        n_points = edge_pairs.max().item() + 1 if edge_pairs.numel() > 0 else 0
        if n_points == 0:
            return torch.tensor([], device=self.device).reshape(0, 3)
        births = torch.zeros(n_points, device=self.device)
        if edge_dists.numel() == 0:
            return torch.tensor([[0.0, 0.0, 0.0] for _ in range(n_points)], device=self.device)
        edge_weights = torch.softmax(-edge_dists, dim=0)
        sorted_indices = torch.argsort(edge_dists)
        sorted_indices = sorted_indices[:self.max_edges]
        component = torch.arange(n_points, device=self.device, dtype=torch.long)
        deaths = []
        for idx in sorted_indices:
            i, j = edge_pairs[idx]
            dist = edge_dists[idx]
            if component[i] != component[j]:
                new_comp = torch.min(component[i], component[j])
                old_comp_i = component[i]
                old_comp_j = component[j]
                new_component = component.clone()
                new_component[component == old_comp_i] = new_comp
                new_component[component == old_comp_j] = new_comp
                component = new_component
                deaths.append(dist)
        if not deaths:
            return torch.tensor([], device=self.device).reshape(0, 3)
        deaths = torch.stack(deaths)
        births = torch.zeros_like(deaths)
        dims = torch.zeros_like(deaths)
        diagram = torch.stack([births, deaths, dims], dim=1)
        diagram = diagram[diagram[:, 1] > diagram[:, 0]]
        return diagram

    def _approximate_dim1(self, edge_dists, edge_pairs):
        n_points = edge_pairs.max().item() + 1 if edge_pairs.numel() > 0 else 0
        if n_points < 3:
            return torch.tensor([], device=self.device).reshape(0, 3)
        sorted_indices = torch.argsort(edge_dists)
        sorted_indices = sorted_indices[:self.max_edges]
        edge_dists_sorted = edge_dists[sorted_indices]
        edge_pairs_sorted = edge_pairs[sorted_indices]
        parent = torch.arange(n_points, device=self.device, dtype=torch.long)
        rank = torch.zeros(n_points, device=self.device, dtype=torch.long)

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return False
            if rank[pu] < rank[pv]:
                pu, pv = pv, pu
            parent[pv] = pu
            if rank[pu] == rank[pv]:
                rank[pu] += 1
            return True

        diagram = []
        cycle_count = 0
        edge_index = 0
        n_edges = len(edge_dists_sorted)
        while edge_index < n_edges and cycle_count < self.max_cycles:
            current_dist = edge_dists_sorted[edge_index]
            cycle_edges = []
            while edge_index < n_edges and abs(edge_dists_sorted[edge_index] - current_dist) < 1e-6:
                i, j = edge_pairs_sorted[edge_index]
                if not union(i, j):
                    cycle_edges.append(edge_index)
                edge_index += 1
            for cycle_idx in cycle_edges:
                birth = edge_dists_sorted[cycle_idx]
                death_idx = edge_index
                while death_idx < n_edges and edge_dists_sorted[death_idx] <= birth:
                    death_idx += 1
                death = edge_dists_sorted[death_idx] if death_idx < n_edges else birth + 0.1
                if death > birth + 0.05:
                    diagram.append([birth, death, 1.0])
                    cycle_count += 1
                    if cycle_count >= self.max_cycles:
                        break
        if not diagram:
            return torch.tensor([], device=self.device).reshape(0, 3)
        diagram = torch.tensor(diagram, device=self.device)
        persistence = diagram[:, 1] - diagram[:, 0]
        weights = torch.sigmoid((persistence - 0.05) * 100)
        diagram[:, :2] = diagram[:, :2] * weights.unsqueeze(1)
        # print(f"Dimension 1 cycles detected: {diagram.shape[0]}")
        return diagram

    def fit_transform(self, X):
        diagrams = []
        for i in range(X.shape[0]):
            points = X[i]
            edge_dists, edge_pairs = self._compute_distance_matrix(points)
            # print(f"Number of edges: {edge_dists.shape[0]}")
            diagram = []
            if 0 in self.homology_dimensions:
                dim0 = self._approximate_dim0(edge_dists, edge_pairs)
                diagram.append(dim0)
            if 1 in self.homology_dimensions:
                dim1 = self._approximate_dim1(edge_dists, edge_pairs)
                diagram.append(dim1)
            if 2 in self.homology_dimensions:
                diagram.append(torch.tensor([], device=self.device).reshape(0, 3))
            diagram = torch.cat(diagram, dim=0) if diagram else torch.tensor([], device=self.device).reshape(0, 3)
            diagrams.append(diagram)
        return diagrams

    def transform(self, X):
        """
        Transform point clouds to persistence diagrams.
        Args:
            X: torch.Tensor of shape (n_samples, n_points, n_features)
        Returns:
            diagrams: List of torch.Tensor, each of shape (n_features, 3)
        """
        return self.fit_transform(X)


def generate_circle(n_points=100, radius=1.0, noise=0.05):
    """
    Generate a 2D point cloud sampling a circle with noise.
    """
    theta = torch.linspace(0, 2 * np.pi, n_points)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y], dim=1)  # Shape: (n_points, 2)
    if noise > 0:
        points += torch.randn_like(points) * noise
    return points


def generate_clusters(n_points=100, n_clusters=3, spread=0.1):
    """
    Generate a 2D point cloud with multiple clusters.
    """
    centers = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])[:n_clusters]
    points = []
    for _ in range(n_points // n_clusters):
        for center in centers:
            point = center + torch.randn(2) * spread
            points.append(point)
    points = torch.stack(points)  # Shape: (n_points, 2)
    return points


class TestDifferentiableVietorisRipsPersistence(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vr = DifferentiableVietorisRipsPersistence(
            homology_dimensions=[0, 1], max_edge_length=0.4, max_edges=200, use_knn=False, auto_config=True
        )
        self.vr_giotto = VietorisRipsPersistence(homology_dimensions=[0, 1])

    def test_circle_diagram(self):
        """Test diagram correctness for a circle (should have dimension 1 features)."""
        points = generate_circle(n_points=100, radius=1.0, noise=0.05).to(self.device)
        points = points.unsqueeze(0)  # Shape: (1, n_points, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]  # Shape: (n_features, 3)

        # Check format
        self.assertEqual(diagram.shape[1], 3, "Diagram should have 3 columns: [birth, death, dim]")
        self.assertTrue(torch.all(diagram[:, 1] >= diagram[:, 0]), "Death times should be >= birth times")

        # Check dimension 1 features
        dim1_mask = diagram[:, 2] == 1
        self.assertGreater(torch.sum(dim1_mask).item(), 0, "Should have dimension 1 features for a circle")
        print(f"Circle diagram: {diagram.shape[0]} features, {torch.sum(dim1_mask).item()} in dim 1")

    def test_clusters_diagram(self):
        """Test diagram correctness for clusters (should have dimension 0 features)."""
        points = generate_clusters(n_points=90, n_clusters=3, spread=0.1).to(self.device)
        points = points.unsqueeze(0)  # Shape: (1, n_points, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]

        # Check format
        self.assertEqual(diagram.shape[1], 3, "Diagram should have 3 columns")
        self.assertTrue(torch.all(diagram[:, 1] >= diagram[:, 0]), "Death times should be >= birth times")

        # Check dimension 0 features
        dim0_mask = diagram[:, 2] == 0
        self.assertGreater(torch.sum(dim0_mask).item(), 0, "Should have dimension 0 features for clusters")
        print(f"Clusters diagram: {diagram.shape[0]} features, {torch.sum(dim0_mask).item()} in dim 0")

    def test_differentiability(self):
        """Test that gradients flow through the computation."""
        points = generate_circle(n_points=100, radius=1.0, noise=0.05).to(self.device)
        points = points.unsqueeze(0).requires_grad_(True)  # Shape: (1, n_points, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]

        # Compute a dummy loss (sum of death times)
        loss = diagram[:, 1].sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(points.grad, "Points should have gradients")
        grad_norm = points.grad.norm().item()
        self.assertGreater(grad_norm, 0, "Gradient norm should be non-zero")
        print(f"Differentiability test: Gradient norm = {grad_norm}")

    def test_comparison_with_giotto(self):
        """Compare diagrams with giotto-tda's VietorisRipsPersistence."""
        points = generate_circle(n_points=100, radius=1.0, noise=0.05)
        points_torch = points.unsqueeze(0).to(self.device)  # Shape: (1, n_points, 2)
        points_numpy = points.numpy()

        # Our implementation
        diagrams_torch = self.vr.fit_transform(points_torch)
        diagram_torch = diagrams_torch[0]

        # giotto-tda
        diagrams_giotto = self.vr_giotto.fit_transform(points_numpy[None])[0]  # Shape: (n_features, 3)
        diagram_giotto = torch.from_numpy(diagrams_giotto).float().to(self.device)  # Cast to float32

        # Compare dimension 1 features
        dim1_torch = diagram_torch[diagram_torch[:, 2] == 1, :2]
        dim1_giotto = diagram_giotto[diagram_giotto[:, 2] == 1, :2]
        if dim1_torch.shape[0] > 0 and dim1_giotto.shape[0] > 0:
            loss, _ = chamfer_distance(dim1_torch.unsqueeze(0), dim1_giotto.unsqueeze(0))
            self.assertLess(loss.item(), 1.0, "Dimension 1 diagrams should be similar")
            print(f"Chamfer distance (dim 1): {loss.item()}")

    def test_empty_input(self):
        """Test handling of empty point cloud."""
        points = torch.empty(1, 0, 2, device=self.device)  # Shape: (1, 0, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]
        self.assertEqual(diagram.shape[0], 0, "Empty input should produce empty diagram")
        print("Empty input test passed")

    def test_small_input(self):
        """Test handling of small point cloud."""
        points = torch.randn(1, 5, 2, device=self.device)  # Shape: (1, 5, 2)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]
        self.assertEqual(diagram.shape[1], 3, "Small input should produce valid diagram")
        print(f"Small input diagram: {diagram.shape[0]} features")

    def test_noisy_input(self):
        """Test robustness to high noise."""
        points = generate_circle(n_points=100, radius=1.0, noise=0.2).to(self.device)
        points = points.unsqueeze(0)
        diagrams = self.vr.fit_transform(points)
        diagram = diagrams[0]
        dim1_mask = diagram[:, 2] == 1
        self.assertGreater(torch.sum(dim1_mask).item(), 0, "Noisy circle should still have dimension 1 features")
        print(f"Noisy input: {torch.sum(dim1_mask).item()} dim 1 features")


def test_on_simple_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)
            self.vr = DifferentiableVietorisRipsPersistence(homology_dimensions=[0, 1, 2], max_edge_length=2.0,
                                                            max_edges=200, use_knn=False, knn_neighbors=20,
                                                            auto_config=True,
                                                            distance_method="optimized_cdist")  # dist_matrix, cdist

        def forward(self, points):
            points = self.fc(points)
            diagrams = self.vr.fit_transform(points)
            batch_features = []
            for i, diagram in enumerate(diagrams):
                print(f"Diagram {i} shape: {diagram.shape}")
                if diagram.shape[0] > 0:
                    print(f"Diagram {i} dim0 features: {(diagram[:, 2] == 0).sum().item()}")
                    print(f"Diagram {i} dim1 features: {(diagram[:, 2] == 1).sum().item()}")
                    print(f"Diagram {i} sample: {diagram[:5]}")
                feature = diagram[:, 0].sum() + diagram[:, 1].sum() \
                    if diagram.shape[0] > 0 else torch.tensor(0.0, device=points.device, requires_grad=True)
                batch_features.append(feature)
            output = torch.stack(batch_features)
            print(f"Output: {output}")
            return output

    def generate_large_circle(n_points=100000, radius=1.0, noise=0.05):
        """
        Generate a 2D point cloud sampling a circle with noise.
        """
        theta = torch.linspace(0, 2 * np.pi, n_points)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        points = torch.stack([x, y], dim=1)  # Shape: (n_points, 2)
        if noise > 0:
            points += torch.randn_like(points) * noise
        return points

    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleModel().to(device)

    # Normalize input to unit scale
    # Generate large point cloud
    points = generate_large_circle(n_points=10000, radius=1.0, noise=0.05).to(device)

    # Prepare input
    points = points.unsqueeze(0).requires_grad_(True)  # Shape: (1, n_subsample, 2)
    points = points / points.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # Normalize

    points.retain_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Forward pass
    output = model(points)
    loss = output.sum()  # Dummy loss
    loss.backward()

    # Check gradients
    print("Model parameter gradients:", model.fc.weight.grad.norm().item())
    print("Input gradients:", points.grad.norm().item())


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)

    # test_on_simple_model()

    # todo: add function to calculate persistent entropy
    # todo: check the function of persistent entropy calculation in torch if equals to the one in package
