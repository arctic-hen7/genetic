use thiserror::Error;

#[derive(Error, Debug)]
pub enum SimulationError {
    #[error(transparent)]
    SetupError(#[from] SimulationSetupError),
}

#[derive(Error, Debug)]
pub enum SimulationSetupError {
    #[error("simulation must have a population updater")]
    NoPopulationUpdater,
    #[error("simulation must have a mutator")]
    NoMutator,
    #[error("simulation must have a crosser")]
    NoCrosser,
    #[error("simulation must have an initial population size")]
    NoPopulationSize,
}
