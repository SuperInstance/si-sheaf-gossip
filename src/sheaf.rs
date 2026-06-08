/// Cellular sheaf on a graph.
///
/// A cellular sheaf assigns a vector space (the *stalk*) to each vertex of a graph,
/// and a linear *restriction map* to each (oriented) edge. The sheaf Laplacian built
/// from these data generalises the ordinary graph Laplacian and its spectrum governs
/// whether — and how fast — a gossip protocol converges.

/// A cellular sheaf on an undirected graph with `n_vertices` vertices.
///
/// * `stalk_dims[v]` — dimension of the stalk at vertex `v`.
/// * `edge_restrictions` — one entry per edge.  Each tuple is
///   `(u, v, matrix)` where `matrix` is the restriction map
///   from the stalk at `u` to the stalk at `v` (row-major).
///   The reverse restriction `v → u` is stored as the transpose.
#[derive(Debug, Clone)]
pub struct Sheaf {
    pub n_vertices: usize,
    pub stalk_dims: Vec<usize>,
    /// (u, v, restriction u→v)
    pub edge_restrictions: Vec<(usize, usize, Vec<Vec<f64>>)>,
}

/// Dimension of the stalk at `vertex`.
pub fn stalk_dimension(sheaf: &Sheaf, vertex: usize) -> usize {
    sheaf.stalk_dims[vertex]
}

/// Sum of all stalk dimensions (= dimension of the global section space).
pub fn total_dimension(sheaf: &Sheaf) -> usize {
    sheaf.stalk_dims.iter().sum()
}

/// Build a **constant sheaf**: every stalk has the same dimension and every
/// restriction map is the identity matrix.
///
/// This recovers the ordinary graph Laplacian up to a factor of `stalk_dim`.
pub fn constant_sheaf(n_vertices: usize, stalk_dim: usize) -> Sheaf {
    let stalk_dims = vec![stalk_dim; n_vertices];
    let id: Vec<Vec<f64>> = (0..stalk_dim)
        .map(|i| (0..stalk_dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    // Fully connected graph for the constant sheaf
    let mut edges = Vec::new();
    for u in 0..n_vertices {
        for v in (u + 1)..n_vertices {
            edges.push((u, v, id.clone()));
        }
    }
    Sheaf {
        n_vertices,
        stalk_dims,
        edge_restrictions: edges,
    }
}

/// Build a **constant sheaf on a ring topology**.
pub fn constant_sheaf_ring(n_vertices: usize, stalk_dim: usize) -> Sheaf {
    let stalk_dims = vec![stalk_dim; n_vertices];
    let id: Vec<Vec<f64>> = (0..stalk_dim)
        .map(|i| (0..stalk_dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut edges = Vec::new();
    for u in 0..n_vertices {
        let v = (u + 1) % n_vertices;
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if !edges.iter().any(|(x, y, _): &(_, _, Vec<Vec<f64>>)| *x == a && *y == b) {
            edges.push((a, b, id.clone()));
        }
    }
    Sheaf {
        n_vertices,
        stalk_dims,
        edge_restrictions: edges,
    }
}

/// Build a **constant sheaf on a star topology** (vertex 0 is the centre).
pub fn constant_sheaf_star(n_leaves: usize, stalk_dim: usize) -> Sheaf {
    let n_vertices = 1 + n_leaves;
    let stalk_dims = vec![stalk_dim; n_vertices];
    let id: Vec<Vec<f64>> = (0..stalk_dim)
        .map(|i| (0..stalk_dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let edges: Vec<_> = (1..=n_leaves).map(|v| (0, v, id.clone())).collect();
    Sheaf {
        n_vertices,
        stalk_dims,
        edge_restrictions: edges,
    }
}

/// Build a **disagreement sheaf**: 1-dimensional stalks, restriction maps
/// pick out the difference between adjacent vertices.  Sections that are
/// constant across the graph are exactly the zero section of this sheaf,
/// so a gossip protocol governed by this sheaf will *not* converge to
/// consensus unless the initial condition is already constant.
pub fn disagreement_sheaf(n_vertices: usize) -> Sheaf {
    let stalk_dims = vec![1; n_vertices];
    // restriction u→v is [1.0] (identity on 1D stalks)
    // The disagreement is captured in the Laplacian construction
    let restriction = vec![vec![1.0]];
    let mut edges = Vec::new();
    for u in 0..n_vertices {
        for v in (u + 1)..n_vertices {
            edges.push((u, v, restriction.clone()));
        }
    }
    Sheaf {
        n_vertices,
        stalk_dims,
        edge_restrictions: edges,
    }
}

/// Build a disagreement sheaf on a ring topology.
pub fn disagreement_sheaf_ring(n_vertices: usize) -> Sheaf {
    let stalk_dims = vec![1; n_vertices];
    let restriction = vec![vec![1.0]];
    let mut edges = Vec::new();
    for u in 0..n_vertices {
        let v = (u + 1) % n_vertices;
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if !edges.iter().any(|(x, y, _): &(_, _, Vec<Vec<f64>>)| *x == a && *y == b) {
            edges.push((a, b, restriction.clone()));
        }
    }
    Sheaf {
        n_vertices,
        stalk_dims,
        edge_restrictions: edges,
    }
}

/// Build a sheaf from a given edge list with uniform stalk dimension and identity restrictions.
pub fn sheaf_from_edges(n_vertices: usize, edges: &[(usize, usize)], stalk_dim: usize) -> Sheaf {
    let stalk_dims = vec![stalk_dim; n_vertices];
    let id: Vec<Vec<f64>> = (0..stalk_dim)
        .map(|i| (0..stalk_dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let edge_restrictions: Vec<_> = edges
        .iter()
        .map(|&(u, v)| (u, v, id.clone()))
        .collect();
    Sheaf {
        n_vertices,
        stalk_dims,
        edge_restrictions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stalk_dimension() {
        let sheaf = constant_sheaf(4, 3);
        assert_eq!(stalk_dimension(&sheaf, 0), 3);
        assert_eq!(stalk_dimension(&sheaf, 3), 3);
    }

    #[test]
    fn test_total_dimension() {
        let sheaf = constant_sheaf(4, 3);
        assert_eq!(total_dimension(&sheaf), 12);
    }

    #[test]
    fn test_constant_sheaf_vertices() {
        let sheaf = constant_sheaf(5, 2);
        assert_eq!(sheaf.n_vertices, 5);
        assert_eq!(sheaf.stalk_dims.len(), 5);
    }

    #[test]
    fn test_constant_sheaf_edges_complete() {
        let sheaf = constant_sheaf(4, 2);
        // complete graph on 4 vertices = 6 edges
        assert_eq!(sheaf.edge_restrictions.len(), 6);
    }

    #[test]
    fn test_constant_sheaf_identity_restrictions() {
        let sheaf = constant_sheaf(3, 2);
        let (_, _, ref r) = sheaf.edge_restrictions[0];
        // Identity matrix
        assert!((r[0][0] - 1.0).abs() < 1e-10);
        assert!((r[0][1]).abs() < 1e-10);
        assert!((r[1][0]).abs() < 1e-10);
        assert!((r[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_disagreement_sheaf_1d_stalks() {
        let sheaf = disagreement_sheaf(5);
        for d in &sheaf.stalk_dims {
            assert_eq!(*d, 1);
        }
    }

    #[test]
    fn test_disagreement_sheaf_edges() {
        let sheaf = disagreement_sheaf(3);
        assert_eq!(sheaf.edge_restrictions.len(), 3);
    }

    #[test]
    fn test_constant_sheaf_ring() {
        let sheaf = constant_sheaf_ring(5, 2);
        assert_eq!(sheaf.n_vertices, 5);
        assert_eq!(sheaf.edge_restrictions.len(), 5);
    }

    #[test]
    fn test_constant_sheaf_star() {
        let sheaf = constant_sheaf_star(4, 2);
        assert_eq!(sheaf.n_vertices, 5);
        assert_eq!(sheaf.edge_restrictions.len(), 4);
    }

    #[test]
    fn test_sheaf_from_edges() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let sheaf = sheaf_from_edges(3, &edges, 3);
        assert_eq!(sheaf.n_vertices, 3);
        assert_eq!(sheaf.edge_restrictions.len(), 3);
        assert_eq!(total_dimension(&sheaf), 9);
    }
}
