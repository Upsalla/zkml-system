"""
Persistent Homology Implementation for TDA-based Model Fingerprinting.

This module implements:
1. Vietoris-Rips complex construction
2. Boundary matrix computation
3. Matrix reduction for persistence computation
4. Persistence diagram extraction
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import heapq


@dataclass
class PersistenceFeature:
    """A single topological feature with birth and death times."""
    dimension: int      # 0 = connected component, 1 = loop, 2 = void
    birth: float        # Filtration value when feature appears
    death: float        # Filtration value when feature disappears (inf if never dies)
    
    @property
    def persistence(self) -> float:
        """Lifetime of the feature."""
        if self.death == float('inf'):
            return float('inf')
        return self.death - self.birth
    
    def __repr__(self):
        return f"H{self.dimension}({self.birth:.4f}, {self.death:.4f})"


@dataclass
class PersistenceDiagram:
    """Collection of persistence features."""
    features: List[PersistenceFeature] = field(default_factory=list)
    
    def add(self, dim: int, birth: float, death: float):
        """Add a feature to the diagram."""
        self.features.append(PersistenceFeature(dim, birth, death))
    
    def top_k(self, k: int, exclude_infinite: bool = True) -> List[PersistenceFeature]:
        """Return k most persistent features."""
        if exclude_infinite:
            finite = [f for f in self.features if f.death != float('inf')]
        else:
            finite = self.features
        return sorted(finite, key=lambda f: -f.persistence)[:k]
    
    def by_dimension(self, dim: int) -> List[PersistenceFeature]:
        """Return features of a specific dimension."""
        return [f for f in self.features if f.dimension == dim]
    
    def summary(self) -> Dict[str, any]:
        """Return summary statistics."""
        dims = defaultdict(list)
        for f in self.features:
            dims[f.dimension].append(f.persistence)
        
        return {
            f"H{d}": {
                "count": len(p),
                "max_persistence": max(p) if p else 0,
                "mean_persistence": np.mean([x for x in p if x != float('inf')]) if p else 0
            }
            for d, p in dims.items()
        }


class SimplexTree:
    """
    Efficient simplex tree for filtration construction.
    
    Stores simplices with their filtration values for efficient
    boundary computation.
    """
    
    def __init__(self):
        self.simplices: Dict[Tuple[int, ...], float] = {}
        self.max_dim = 0
    
    def insert(self, simplex: Tuple[int, ...], filtration: float):
        """Insert a simplex with its filtration value."""
        simplex = tuple(sorted(simplex))
        if simplex not in self.simplices or self.simplices[simplex] > filtration:
            self.simplices[simplex] = filtration
            self.max_dim = max(self.max_dim, len(simplex) - 1)
    
    def get_filtration(self, simplex: Tuple[int, ...]) -> float:
        """Get filtration value of a simplex."""
        return self.simplices.get(tuple(sorted(simplex)), float('inf'))
    
    def get_simplices_by_dim(self, dim: int) -> List[Tuple[Tuple[int, ...], float]]:
        """Get all simplices of a given dimension, sorted by filtration."""
        result = [(s, f) for s, f in self.simplices.items() if len(s) == dim + 1]
        return sorted(result, key=lambda x: (x[1], x[0]))
    
    def boundary(self, simplex: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Compute boundary of a simplex (faces of dimension d-1)."""
        if len(simplex) <= 1:
            return []
        return [tuple(simplex[:i] + simplex[i+1:]) for i in range(len(simplex))]


class VietorisRipsComplex:
    """
    Constructs Vietoris-Rips complex from a point cloud.
    
    For a set of points P and threshold ε, the VR complex contains:
    - A vertex for each point
    - An edge for each pair of points with distance ≤ ε
    - A k-simplex for each (k+1) points that are pairwise within distance ε
    """
    
    def __init__(self, points: np.ndarray, max_dim: int = 1, max_edge_length: float = None):
        """
        Initialize VR complex.
        
        Args:
            points: (N, D) array of N points in D dimensions
            max_dim: Maximum simplex dimension to compute
            max_edge_length: Maximum edge length to consider (default: diameter of point cloud)
        """
        self.points = points
        self.n_points = len(points)
        self.max_dim = max_dim
        
        # Compute pairwise distances
        self.distances = self._compute_distances()
        
        if max_edge_length is None:
            max_edge_length = np.max(self.distances)
        self.max_edge_length = max_edge_length
        
        # Build simplex tree
        self.simplex_tree = self._build_filtration()
    
    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        n = self.n_points
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.points[i] - self.points[j])
                distances[i, j] = d
                distances[j, i] = d
        return distances
    
    def _build_filtration(self) -> SimplexTree:
        """Build the filtration (sequence of growing complexes)."""
        tree = SimplexTree()
        
        # Add vertices at filtration 0
        for i in range(self.n_points):
            tree.insert((i,), 0.0)
        
        # Add edges at their distance
        edges = []
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                d = self.distances[i, j]
                if d <= self.max_edge_length:
                    edges.append((d, i, j))
        
        # Sort edges by distance
        edges.sort()
        
        for d, i, j in edges:
            tree.insert((i, j), d)
        
        # Add higher-dimensional simplices if needed
        if self.max_dim >= 2:
            self._add_higher_simplices(tree)
        
        return tree
    
    def _add_higher_simplices(self, tree: SimplexTree):
        """Add triangles and higher simplices."""
        # For each triple of vertices, check if they form a 2-simplex
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                for k in range(j + 1, self.n_points):
                    # Check if all edges exist
                    d_ij = self.distances[i, j]
                    d_jk = self.distances[j, k]
                    d_ik = self.distances[i, k]
                    
                    if max(d_ij, d_jk, d_ik) <= self.max_edge_length:
                        # Add triangle at max edge length
                        tree.insert((i, j, k), max(d_ij, d_jk, d_ik))


class PersistenceComputer:
    """
    Computes persistent homology using matrix reduction.
    
    Uses the standard algorithm:
    1. Build boundary matrix
    2. Reduce to column echelon form
    3. Read off persistence pairs
    """
    
    def __init__(self, simplex_tree: SimplexTree):
        self.simplex_tree = simplex_tree
    
    def compute(self, max_dim: int = 1) -> PersistenceDiagram:
        """
        Compute persistence diagram up to given dimension.
        
        Args:
            max_dim: Maximum homology dimension to compute
            
        Returns:
            PersistenceDiagram with all features
        """
        diagram = PersistenceDiagram()
        
        # Get all simplices sorted by filtration
        all_simplices = []
        for dim in range(max_dim + 2):  # Need dim+1 simplices for H_dim
            all_simplices.extend(self.simplex_tree.get_simplices_by_dim(dim))
        
        # Sort by (filtration, dimension, simplex)
        all_simplices.sort(key=lambda x: (x[1], len(x[0]), x[0]))
        
        # Create index mapping
        simplex_to_idx = {s: i for i, (s, _) in enumerate(all_simplices)}
        n = len(all_simplices)
        
        # Build boundary matrix (sparse representation)
        # boundary[j] = set of indices i where ∂_j has a 1 in row i
        boundary = [set() for _ in range(n)]
        
        for j, (simplex, _) in enumerate(all_simplices):
            for face in self.simplex_tree.boundary(simplex):
                if face in simplex_to_idx:
                    boundary[j].add(simplex_to_idx[face])
        
        # Reduce matrix (standard persistence algorithm)
        low = {}  # low[j] = lowest 1 in column j after reduction
        
        for j in range(n):
            while boundary[j]:
                i = max(boundary[j])
                if i in low:
                    # Column collision - add columns
                    k = low[i]
                    boundary[j] = boundary[j].symmetric_difference(boundary[k])
                else:
                    low[i] = j
                    break
        
        # Extract persistence pairs
        paired = set()
        
        for i, j in low.items():
            paired.add(i)
            paired.add(j)
            
            simplex_i, filt_i = all_simplices[i]
            simplex_j, filt_j = all_simplices[j]
            
            dim = len(simplex_i) - 1  # Dimension of the dying feature
            
            if filt_i < filt_j:  # Only add if birth < death
                diagram.add(dim, filt_i, filt_j)
        
        # Add unpaired simplices (features that never die)
        for i, (simplex, filt) in enumerate(all_simplices):
            if i not in paired:
                dim = len(simplex) - 1
                if dim <= max_dim:
                    diagram.add(dim, filt, float('inf'))
        
        return diagram


def compute_persistence(points: np.ndarray, max_dim: int = 1, 
                       max_edge_length: float = None) -> PersistenceDiagram:
    """
    Convenience function to compute persistence diagram from points.
    
    Args:
        points: (N, D) array of N points in D dimensions
        max_dim: Maximum homology dimension
        max_edge_length: Maximum edge length (default: auto)
        
    Returns:
        PersistenceDiagram
    """
    vr = VietorisRipsComplex(points, max_dim, max_edge_length)
    computer = PersistenceComputer(vr.simplex_tree)
    return computer.compute(max_dim)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Persistent Homology Tests")
    print("=" * 60)
    
    # Test 1: Simple triangle
    print("\n1. Testing with a triangle")
    triangle = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ])
    
    diagram = compute_persistence(triangle, max_dim=1)
    print(f"   Points: {len(triangle)}")
    print(f"   Features: {diagram.features}")
    print(f"   Summary: {diagram.summary()}")
    
    # Test 2: Square with a hole (should detect H1)
    print("\n2. Testing with a square (should have H1 feature)")
    square = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])
    
    diagram = compute_persistence(square, max_dim=1, max_edge_length=1.5)
    print(f"   Points: {len(square)}")
    print(f"   Features: {diagram.features}")
    print(f"   H0 features: {diagram.by_dimension(0)}")
    print(f"   H1 features: {diagram.by_dimension(1)}")
    
    # Test 3: Random point cloud
    print("\n3. Testing with random point cloud")
    np.random.seed(42)
    random_points = np.random.randn(20, 3)
    
    diagram = compute_persistence(random_points, max_dim=1, max_edge_length=2.0)
    print(f"   Points: {len(random_points)}")
    print(f"   Total features: {len(diagram.features)}")
    print(f"   Top 5 by persistence: {diagram.top_k(5)}")
    print(f"   Summary: {diagram.summary()}")
    
    # Test 4: Circle (should have strong H1)
    print("\n4. Testing with circle (should have strong H1)")
    n_circle = 16
    theta = np.linspace(0, 2*np.pi, n_circle, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    diagram = compute_persistence(circle, max_dim=1, max_edge_length=1.0)
    print(f"   Points: {len(circle)}")
    print(f"   H1 features: {diagram.by_dimension(1)}")
    print(f"   Most persistent H1: {max(diagram.by_dimension(1), key=lambda f: f.persistence, default=None)}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
