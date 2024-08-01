use serde::{Deserialize, Serialize};
use serde_json::error::Error;
use std::fs;

#[derive(Serialize, Deserialize, Debug)]
pub struct CommonParam {
    pub dt: f64,
    pub simulation_time: f64,
    pub random_range: f64,
    pub random_seeds: Vec<u64>,
}

impl CommonParam {
    pub fn load() -> Result<Self, Error> {
        serde_json::from_str(&fs::read_to_string("data/common.json").unwrap())
    }
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

impl NetworkParam {
    pub fn load(path: &str) -> Result<Self, Error> {
        serde_json::from_str(&fs::read_to_string(path).unwrap())
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ControlParam {
    comment: String,
    pub control_type: usize,
    pub input_weight: Vec<f64>,
    pub input_frequency: Vec<f64>,
    pub average_weight: Vec<Vec<f64>>,
}

impl ControlParam {
    pub fn load(path: &str) -> Result<Self, Error> {
        serde_json::from_str(&fs::read_to_string(path).unwrap())
    }
}
