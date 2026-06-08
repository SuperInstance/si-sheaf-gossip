pub mod sheaf;
pub mod laplacian;
pub mod gossip;
pub mod experiment;

use experiment::summary;

/// Print experiment summary to stdout.
pub fn run_experiments() {
    println!("{}", summary());
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::sheaf::*;
    use crate::laplacian::*;
    use crate::gossip::*;

    #[test]
    fn test_full_pipeline_constant_sheaf() {
        let sheaf = constant_sheaf(5, 2);
        let gap = spectral_gap(&sheaf);
        assert!(gap > 0.0);

        let config = GossipConfig {
            n_agents: 5,
            topology: Topology::Complete(5),
            sheaf: sheaf.clone(),
            step_size: 0.05,
            noise: 0.0,
        };
        let initial: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, (5 - i) as f64]).collect();
        let history = run_gossip(&initial, &config, 200);
        let errors = consensus_error(&history);
        assert!(errors[200] < errors[0]);
    }

    #[test]
    fn test_full_pipeline_disagreement_sheaf() {
        let sheaf = disagreement_sheaf(4);
        let lap = sheaf_laplacian(&sheaf);
        let eigs = eigenvalues_symmetric(&lap);
        // First eigenvalue should be ~0
        assert!(eigs[0].abs() < 1e-6);
        // Should have positive eigenvalues
        assert!(eigs.last().unwrap() > &0.0);
    }

    #[test]
    fn test_ring_vs_complete_integration() {
        let n = 6;
        let sheaf_ring = constant_sheaf_ring(n, 1);
        let sheaf_complete = constant_sheaf(n, 1);

        let gap_ring = spectral_gap(&sheaf_ring);
        let gap_complete = spectral_gap(&sheaf_complete);

        assert!(gap_complete > gap_ring, "Complete graph gap > Ring gap");
    }

    #[test]
    fn test_summary_completes() {
        let s = summary();
        assert!(s.len() > 100);
    }

    #[test]
    fn test_sheaf_laplacian_matches_graph_laplacian() {
        // For constant sheaf with stalk_dim=1, sheaf Laplacian should equal graph Laplacian
        let n = 4;
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|u| (u + 1..n).map(move |v| (u, v)))
            .collect();
        let graph_lap = graph_laplacian(n, &edges);
        let sheaf = constant_sheaf(n, 1);
        let sheaf_lap = sheaf_laplacian(&sheaf);

        // They should be equal (both use complete graph on n vertices)
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (graph_lap[i][j] - sheaf_lap[i][j]).abs() < 1e-10,
                    "Mismatch at ({}, {}): graph={}, sheaf={}",
                    i, j, graph_lap[i][j], sheaf_lap[i][j]
                );
            }
        }
    }

    #[test]
    fn test_run_experiments_no_panic() {
        run_experiments();
    }
}
