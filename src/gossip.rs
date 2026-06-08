/// Gossip protocol simulation on a sheaf.
///
/// Agents sit at graph vertices, each holding a vector in their stalk.
/// At each gossip step, every agent averages its state with its
/// neighbours' states, weighted by the sheaf Laplacian.  The spectral
/// gap of the sheaf Laplacian predicts how many rounds are needed to
/// reach consensus.

use crate::sheaf::Sheaf;
use crate::laplacian::sheaf_laplacian;

/// Network topology for the gossip simulation.
#[derive(Debug, Clone)]
pub enum Topology {
    /// Ring: each vertex connected to its two neighbours.
    Ring(usize),
    /// Complete: every pair connected.
    Complete(usize),
    /// Star: vertex 0 is the hub.
    Star(usize),
    /// Erdős–Rényi random graph with `n` vertices and edge probability `p`.
    ErdosRenyi(usize, f64),
}

/// Configuration for a gossip simulation.
#[derive(Debug, Clone)]
pub struct GossipConfig {
    pub n_agents: usize,
    pub topology: Topology,
    pub sheaf: Sheaf,
    pub step_size: f64,
    pub noise: f64,
}

/// Generate an edge list from a topology description.
pub fn generate_topology(topo: &Topology) -> Vec<(usize, usize)> {
    match topo {
        Topology::Ring(n) => {
            let mut edges = Vec::new();
            for i in 0..*n {
                let j = (i + 1) % *n;
                let (u, v) = if i < j { (i, j) } else { (j, i) };
                if !edges.contains(&(u, v)) {
                    edges.push((u, v));
                }
            }
            edges
        }
        Topology::Complete(n) => {
            let mut edges = Vec::new();
            for u in 0..*n {
                for v in (u + 1)..*n {
                    edges.push((u, v));
                }
            }
            edges
        }
        Topology::Star(n_leaves) => {
            let n = *n_leaves;
            (1..=n).map(|v| (0, v)).collect()
        }
        Topology::ErdosRenyi(n, p) => {
            let mut edges = Vec::new();
            // Use a simple deterministic PRNG for reproducibility
            let mut seed: u64 = 42;
            for u in 0..*n {
                for v in (u + 1)..*n {
                    // xorshift64
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let rand_val = (seed & 0xFFFF) as f64 / 65536.0;
                    if rand_val < *p {
                        edges.push((u, v));
                    }
                }
            }
            edges
        }
    }
}

/// Perform one gossip step.
///
/// `states[v]` is the current state vector of agent `v`.  The update rule is:
///
/// ```text
/// states_new = states - step_size * L_F * states
/// ```
///
/// where `L_F` is the sheaf Laplacian.  Optional Gaussian noise can be added.
pub fn gossip_step(
    states: &[Vec<f64>],
    sheaf: &Sheaf,
    config: &GossipConfig,
) -> Vec<Vec<f64>> {
    let lap = sheaf_laplacian(sheaf);

    // Flatten states into a single vector
    let mut flat: Vec<f64> = Vec::new();
    for s in states {
        flat.extend_from_slice(s);
    }

    // Compute L_F * states
    let n = flat.len();
    let mut product = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            product[i] += lap[i][j] * flat[j];
        }
    }

    // states_new = states - step_size * product + noise
    let mut result_flat = vec![0.0; n];
    // Simple deterministic noise (seeded)
    let mut noise_seed: u64 = 12345;
    for i in 0..n {
        noise_seed ^= noise_seed << 13;
        noise_seed ^= noise_seed >> 7;
        noise_seed ^= noise_seed << 17;
        let noise_val = if config.noise > 0.0 {
            // Box-Muller-ish approximation
            let u = (noise_seed & 0xFFFF) as f64 / 65536.0;
            config.noise * (2.0 * std::f64::consts::PI * u).cos() * ((-2.0 * u.ln().max(1e-30)).min(10.0)).sqrt()
        } else {
            0.0
        };
        result_flat[i] = flat[i] - config.step_size * product[i] + noise_val;
    }

    // Unflatten
    let mut result = Vec::new();
    let mut idx = 0;
    for v in 0..sheaf.n_vertices {
        let dim = sheaf.stalk_dims[v];
        result.push(result_flat[idx..idx + dim].to_vec());
        idx += dim;
    }
    result
}

/// Run a full gossip simulation for `n_steps` rounds.
///
/// Returns the full history of states.
pub fn run_gossip(
    initial: &[Vec<f64>],
    config: &GossipConfig,
    n_steps: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut history = Vec::new();
    let mut states = initial.to_vec();
    history.push(states.clone());
    for _ in 0..n_steps {
        states = gossip_step(&states, &config.sheaf, config);
        history.push(states.clone());
    }
    history
}

/// Compute consensus error at each step: average L2 distance from the mean.
pub fn consensus_error(history: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let n_agents = history[0].len();
    let dim = history[0][0].len();
    history
        .iter()
        .map(|states| {
            // Compute mean
            let mut mean = vec![0.0; dim];
            for s in states {
                for k in 0..dim {
                    mean[k] += s[k];
                }
            }
            for m in &mut mean {
                *m /= n_agents as f64;
            }
            // Average distance from mean
            let mut total_dist = 0.0;
            for s in states {
                let d: f64 = s.iter()
                    .zip(mean.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                total_dist += d.sqrt();
            }
            total_dist / n_agents as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheaf::*;

    #[test]
    fn test_generate_ring_topology() {
        let edges = generate_topology(&Topology::Ring(5));
        assert_eq!(edges.len(), 5);
    }

    #[test]
    fn test_generate_complete_topology() {
        let edges = generate_topology(&Topology::Complete(4));
        assert_eq!(edges.len(), 6);
    }

    #[test]
    fn test_generate_star_topology() {
        let edges = generate_topology(&Topology::Star(3));
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_generate_erdos_renyi() {
        let edges = generate_topology(&Topology::ErdosRenyi(10, 0.5));
        assert!(!edges.is_empty());
        assert!(edges.len() <= 45); // max for 10 vertices
    }

    #[test]
    fn test_gossip_step_constant_sheaf() {
        let sheaf = constant_sheaf(3, 1);
        let config = GossipConfig {
            n_agents: 3,
            topology: Topology::Complete(3),
            sheaf: sheaf.clone(),
            step_size: 0.01,
            noise: 0.0,
        };
        let initial = vec![vec![1.0], vec![2.0], vec![3.0]];
        let new_states = gossip_step(&initial, &sheaf, &config);
        assert_eq!(new_states.len(), 3);
        // States should move toward the mean (2.0)
        assert!(new_states[0][0] > 1.0);
        assert!(new_states[2][0] < 3.0);
    }

    #[test]
    fn test_gossip_convergence_constant_sheaf() {
        let sheaf = constant_sheaf(4, 1);
        let config = GossipConfig {
            n_agents: 4,
            topology: Topology::Complete(4),
            sheaf: sheaf.clone(),
            step_size: 0.05,
            noise: 0.0,
        };
        let initial = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let history = run_gossip(&initial, &config, 200);
        let errors = consensus_error(&history);
        assert!(errors[200] < errors[0], "Error should decrease");
        assert!(errors[200] < 0.1, "Should converge: final error = {}", errors[200]);
    }

    #[test]
    fn test_consensus_error_initial() {
        let states = vec![vec![0.0], vec![2.0]];
        let errors = consensus_error(&[states]);
        assert!((errors[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_consensus_error_zero_at_consensus() {
        let states = vec![vec![1.5], vec![1.5], vec![1.5]];
        let errors = consensus_error(&[states]);
        assert!(errors[0] < 1e-10);
    }

    #[test]
    fn test_ring_convergence_slower() {
        let sheaf_ring = constant_sheaf_ring(6, 1);
        let sheaf_complete = constant_sheaf(6, 1);

        let config_ring = GossipConfig {
            n_agents: 6,
            topology: Topology::Ring(6),
            sheaf: sheaf_ring,
            step_size: 0.05,
            noise: 0.0,
        };
        let config_complete = GossipConfig {
            n_agents: 6,
            topology: Topology::Complete(6),
            sheaf: sheaf_complete,
            step_size: 0.05,
            noise: 0.0,
        };

        let initial: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
        let err_ring = consensus_error(&run_gossip(&initial, &config_ring, 100));
        let err_complete = consensus_error(&run_gossip(&initial, &config_complete, 100));

        // Complete graph should converge faster
        assert!(err_complete[100] < err_ring[100],
            "Complete should converge faster: {} vs {}", err_complete[100], err_ring[100]);
    }

    #[test]
    fn test_multidim_gossip() {
        let sheaf = constant_sheaf(3, 3);
        let config = GossipConfig {
            n_agents: 3,
            topology: Topology::Complete(3),
            sheaf: sheaf.clone(),
            step_size: 0.05,
            noise: 0.0,
        };
        let initial = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let history = run_gossip(&initial, &config, 200);
        let errors = consensus_error(&history);
        assert!(errors[200] < errors[0]);
    }

    #[test]
    fn test_gossip_preserves_mean_constant_sheaf() {
        let sheaf = constant_sheaf(4, 1);
        let config = GossipConfig {
            n_agents: 4,
            topology: Topology::Complete(4),
            sheaf: sheaf.clone(),
            step_size: 0.05,
            noise: 0.0,
        };
        let initial = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let initial_mean: f64 = initial.iter().map(|v| v[0]).sum::<f64>() / 4.0;
        let history = run_gossip(&initial, &config, 100);
        for step in &[0, 50, 100] {
            let mean: f64 = history[*step].iter().map(|v| v[0]).sum::<f64>() / 4.0;
            assert!((mean - initial_mean).abs() < 0.01,
                "Mean not preserved at step {}: {} vs {}", step, mean, initial_mean);
        }
    }
}
