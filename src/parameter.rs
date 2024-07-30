use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct CommonParam {
    pub dt: f64,
    pub simulation_time: f64,
    pub random_range: f64,
    pub random_seeds: Vec<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NetworkParam {
    comment: String,
    pub state_dim: usize,
    pub input_dim: usize,
    pub cluster_nodes_num: Vec<usize>,
    pub frequency: Vec<f64>,
    pub connectivity: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ControlParam {
    comment: String,
    pub control_type: usize,
    pub input_weight: Vec<f64>,
    pub input_frequency: Vec<f64>,
    pub average_weight: Vec<Vec<f64>>,
}
