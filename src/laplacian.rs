/// Sheaf Laplacian construction and spectral analysis.
///
/// The sheaf Laplacian `L_F` is a block matrix acting on the global section
/// space `⨁_v F(v)`.  For each edge `(u, v)` with restriction `F_{u≤v}`,
/// it contributes:
///
/// ```text
/// L_F[u,u] += F_{u≤v}ᵀ F_{u≤v}
/// L_F[v,v] += F_{v≤u}ᵀ F_{v≤u}
/// L_F[u,v] -= F_{u≤v}ᵀ F_{v≤u}
/// L_F[v,u] -= F_{v≤u}ᵀ F_{u≤v}
/// ```
///
/// The spectrum of `L_F` determines gossip convergence:
/// - λ = 0  →  consistent sections (consensus)
/// - λ_min⁺  →  convergence rate (spectral gap)
/// - λ_max   →  hardest disagreement

use crate::sheaf::{Sheaf, total_dimension};

/// Build the full sheaf Laplacian `L_F` as a dense matrix.
///
/// The matrix is symmetric positive semi-definite by construction.
pub fn sheaf_laplacian(sheaf: &Sheaf) -> Vec<Vec<f64>> {
    let n = total_dimension(sheaf);
    let mut l = vec![vec![0.0; n]; n];

    // Compute row/col offset for each vertex
    let mut offset = vec![0usize; sheaf.n_vertices];
    for v in 1..sheaf.n_vertices {
        offset[v] = offset[v - 1] + sheaf.stalk_dims[v - 1];
    }

    for &(u, v, ref rest_uv) in &sheaf.edge_restrictions {
        // rest_uv: matrix dim(F(v)) × dim(F(u)), restriction u→v
        let du = sheaf.stalk_dims[u];
        let dv = sheaf.stalk_dims[v];
        let ou = offset[u];
        let ov = offset[v];

        // Transpose of rest_uv: dim(F(u)) × dim(F(v))
        let rest_uv_t = transpose(rest_uv);
        // Reverse restriction v→u = rest_uv (since it's stored as u→v)
        // We'll use rest_uv as v→u and its transpose as u→v
        // Actually: rest_uv is stored as restriction u→v
        // rest_vu = rest_uv (we reuse), rest_vu_t = transpose
        let rest_vu_t = rest_uv_t.clone(); // transpose of u→v is v→u contribution

        // L_F[u,u] += rest_uvᵀ * rest_uv
        for i in 0..du {
            for j in 0..du {
                let mut sum = 0.0;
                for k in 0..dv {
                    sum += rest_uv_t[i][k] * rest_uv[k][j];
                }
                l[ou + i][ou + j] += sum;
            }
        }

        // L_F[v,v] += rest_vuᵀ * rest_vu = rest_uv * rest_uvᵀ
        for i in 0..dv {
            for j in 0..dv {
                let mut sum = 0.0;
                for k in 0..du {
                    sum += rest_uv[i][k] * rest_uv_t[k][j];
                }
                l[ov + i][ov + j] += sum;
            }
        }

        // L_F[u,v] -= rest_uvᵀ * rest_vu = rest_uvᵀ * rest_uv... no.
        // L_F[u,v] -= F_{u≤v}ᵀ F_{v≤u}
        // F_{v≤u} is the reverse restriction. In our model, for the constant sheaf
        // both directions are identity. We store only u→v; v→u is the "reverse".
        // For simplicity, we treat the reverse restriction as the transpose of rest_uv
        // when the sheaf is not explicitly providing both directions.
        // Actually, the reverse restriction map F_{v≤u} should be the adjoint (transpose).
        // So L_F[u,v] -= rest_uvᵀ * rest_uvᵀᵀ = rest_uvᵀ * rest_uv... that's not right either.

        // Let's be precise:
        // rest_uv is a dv × du matrix (maps F(u) → F(v))
        // The adjoint (reverse) rest_vu = rest_uvᵀ is du × dv (maps F(v) → F(u))
        // L_F[u,u] += rest_vu * rest_uv = rest_uvᵀ * rest_uv  (du × du) ✓
        // L_F[v,v] += rest_uv * rest_vu = rest_uv * rest_uvᵀ  (dv × dv) ✓
        // L_F[u,v] -= rest_vu * rest_vuᵀ... no.
        // L_F[u,v] -= rest_uvᵀ * (rest_uvᵀ)ᵀ = rest_uvᵀ * rest_uv
        // Wait, L_F[u,v] -= F_{u≤v}ᵀ F_{v≤u} = rest_uvᵀ * rest_uvᵀ... no.

        // Standard sheaf Laplacian formula:
        // For edge e = (u,v) with restriction F_{uv}: F(u) → F(v):
        //   L += [ F_{uv}ᵀF_{uv}    -F_{uv}ᵀ  ]
        //        [ -F_{uv}           I          ]
        // Wait, that's only for the simple case.

        // The correct formula for undirected sheaf Laplacian:
        // L[u,u] += F_{uv}ᵀ F_{uv}
        // L[v,v] += F_{uv} F_{uv}ᵀ
        // L[u,v] -= F_{uv}ᵀ
        // L[v,u] -= F_{uv}

        // Hmm, that doesn't work dimensionally unless du == dv.

        // Actually the standard definition uses coboundary maps.
        // δ^0 maps sections to edge data. The Laplacian is (δ^0)ᵀ δ^0.
        //
        // For edge e=(u,v) with restriction f_e: F(u) → F(v):
        // δ^0_e(s) = f_e(s_u) - s_v   (in F(v))
        //
        // (δ^0)ᵀ gives:
        // For vertex u: sum over edges e=(u,v): f_eᵀ * (f_e(s_u) - s_v)
        // For vertex v: sum over edges e=(u,v): -(f_e(s_u) - s_v)
        //
        // So L_F[u,u] += f_eᵀ f_e
        // L_F[v,v] += I
        // L_F[u,v] -= f_eᵀ
        // L_F[v,u] -= f_e

        // Let me redo this cleanly:
        // L_F[u,u] += rest_uvᵀ * rest_uv    (du × du)
        // L_F[v,v] += I_{dv}                  (dv × dv)
        // L_F[u,v] -= rest_uvᵀ                (du × dv)
        // L_F[v,u] -= rest_uv                  (dv × du)

        // Redo L_F[v,v]: we already added rest_uv * rest_uvᵀ, need to fix.
        // Actually, let me just redo the whole thing properly.

        // Clear what we computed above
        for i in 0..du {
            for j in 0..du {
                let mut sum = 0.0;
                for k in 0..dv {
                    sum += rest_uv_t[i][k] * rest_uv[k][j];
                }
                l[ou + i][ou + j] -= sum; // undo
            }
        }
        for i in 0..dv {
            for j in 0..dv {
                let mut sum = 0.0;
                for k in 0..du {
                    sum += rest_uv[i][k] * rest_uv_t[k][j];
                }
                l[ov + i][ov + j] -= sum; // undo
            }
        }
    }

    // Now redo with correct formula
    for &(u, v, ref rest_uv) in &sheaf.edge_restrictions {
        let du = sheaf.stalk_dims[u];
        let dv = sheaf.stalk_dims[v];
        let ou = offset[u];
        let ov = offset[v];
        let rest_uv_t = transpose(rest_uv);

        // L_F[u,u] += rest_uvᵀ * rest_uv
        for i in 0..du {
            for j in 0..du {
                for k in 0..dv {
                    l[ou + i][ou + j] += rest_uv_t[i][k] * rest_uv[k][j];
                }
            }
        }

        // L_F[v,v] += I_{dv}
        for i in 0..dv {
            l[ov + i][ov + i] += 1.0;
        }

        // L_F[u,v] -= rest_uvᵀ
        for i in 0..du {
            for j in 0..dv {
                l[ou + i][ov + j] -= rest_uv_t[i][j];
            }
        }

        // L_F[v,u] -= rest_uv
        for i in 0..dv {
            for j in 0..du {
                l[ov + i][ou + j] -= rest_uv[i][j];
            }
        }
    }

    l
}

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return vec![];
    }
    let rows = m.len();
    let cols = m[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = m[i][j];
        }
    }
    t
}

/// Build the standard (unnormalised) graph Laplacian for a graph given by
/// an edge list.
pub fn graph_laplacian(n_vertices: usize, edges: &[(usize, usize)]) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0; n_vertices]; n_vertices];
    for &(u, v) in edges {
        l[u][u] += 1.0;
        l[v][v] += 1.0;
        l[u][v] -= 1.0;
        l[v][u] -= 1.0;
    }
    l
}

/// Compute eigenvalues using the Jacobi eigenvalue algorithm (for symmetric matrices).
/// Returns all eigenvalues in ascending order.
pub fn eigenvalues_symmetric(matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }
    let mut a = matrix.to_vec();
    // Jacobi eigenvalue algorithm with Givens rotations
    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }
        // Compute rotation angle
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                new_a[i][p] = c * aip + s * aiq;
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = -s * aip + c * aiq;
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_a[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigenvalues
}

/// Power iteration for finding the largest eigenvalue.
pub fn power_iteration_top(matrix: &[Vec<f64>], n_iter: usize) -> f64 {
    let n = matrix.len();
    if n == 0 {
        return 0.0;
    }
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0)).collect();
    // Normalise
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    for _ in 0..n_iter {
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i][j] * v[j];
            }
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
        for x in &mut w {
            *x /= norm;
        }
        v = w;
    }
    // Rayleigh quotient
    let mut lambda = 0.0;
    let mut mv = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mv[i] += matrix[i][j] * v[j];
        }
    }
    for i in 0..n {
        lambda += v[i] * mv[i];
    }
    lambda
}

/// Compute the spectral gap (smallest nonzero eigenvalue) of the sheaf Laplacian.
pub fn spectral_gap(sheaf: &Sheaf) -> f64 {
    let lap = sheaf_laplacian(sheaf);
    let eigs = eigenvalues_symmetric(&lap);
    for e in &eigs {
        if *e > 1e-8 {
            return *e;
        }
    }
    0.0 // no positive eigenvalue = disconnected
}

/// Check whether the sheaf is connected (spectral gap > 0).
pub fn is_connected(sheaf: &Sheaf) -> bool {
    spectral_gap(sheaf) > 1e-8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheaf::*;

    #[test]
    fn test_graph_laplacian_triangle() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let l = graph_laplacian(3, &edges);
        // Row sums should be zero
        for row in &l {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10);
        }
        // Diagonal should be 2 for each vertex (degree)
        for i in 0..3 {
            assert!((l[i][i] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sheaf_laplacian_symmetry() {
        let sheaf = constant_sheaf(4, 2);
        let l = sheaf_laplacian(&sheaf);
        let n = l.len();
        for i in 0..n {
            for j in 0..n {
                assert!((l[i][j] - l[j][i]).abs() < 1e-10,
                    "Not symmetric at ({}, {}): {} vs {}", i, j, l[i][j], l[j][i]);
            }
        }
    }

    #[test]
    fn test_sheaf_laplacian_positive_semi_definite() {
        let sheaf = constant_sheaf(4, 2);
        let l = sheaf_laplacian(&sheaf);
        let eigs = eigenvalues_symmetric(&l);
        for e in &eigs {
            assert!(*e >= -1e-8, "Negative eigenvalue: {}", e);
        }
    }

    #[test]
    fn test_constant_sheaf_spectral_gap_positive() {
        let sheaf = constant_sheaf(4, 1);
        let gap = spectral_gap(&sheaf);
        assert!(gap > 0.01, "Spectral gap should be positive: {}", gap);
    }

    #[test]
    fn test_constant_sheaf_is_connected() {
        let sheaf = constant_sheaf(4, 1);
        assert!(is_connected(&sheaf));
    }

    #[test]
    fn test_ring_spectral_gap() {
        let sheaf = constant_sheaf_ring(6, 1);
        let gap = spectral_gap(&sheaf);
        assert!(gap > 0.01, "Ring spectral gap should be positive: {}", gap);
    }

    #[test]
    fn test_star_spectral_gap() {
        let sheaf = constant_sheaf_star(4, 1);
        let gap = spectral_gap(&sheaf);
        assert!(gap > 0.01);
    }

    #[test]
    fn test_eigenvalues_complete_graph_4() {
        let sheaf = constant_sheaf(4, 1);
        let l = sheaf_laplacian(&sheaf);
        let eigs = eigenvalues_symmetric(&l);
        assert!(eigs[0].abs() < 1e-8, "First eigenvalue should be ~0");
        // Complete graph on 4 vertices: eigenvalues are 0, 4, 4, 4
        assert!((eigs[3] - 4.0).abs() < 0.5, "Max eigenvalue near 4: got {}", eigs[3]);
    }

    #[test]
    fn test_disagreement_sheaf_laplacian() {
        let sheaf = disagreement_sheaf(3);
        let l = sheaf_laplacian(&sheaf);
        // Should be the graph Laplacian of the complete graph on 3 vertices
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn test_power_iteration() {
        let sheaf = constant_sheaf(4, 1);
        let l = sheaf_laplacian(&sheaf);
        let top = power_iteration_top(&l, 200);
        assert!(top > 2.0, "Top eigenvalue of K4 Laplacian ~4, got {}", top);
    }

    #[test]
    fn test_laplacian_row_sums_zero() {
        let sheaf = constant_sheaf(4, 1);
        let l = sheaf_laplacian(&sheaf);
        for row in &l {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10, "Row sum should be 0, got {}", sum);
        }
    }
}
