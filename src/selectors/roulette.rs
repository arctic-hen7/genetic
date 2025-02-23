use super::ProbabilitySelector;
use crate::population::{Individual, Population};
use ordered_float::NotNan;
use std::{iter::Sum, ops::Div};

/// A selector that assigns probabilities of selection to individuals by their fitness scores,
/// directly making those individuals with higher fitness have linearly proportional selective
/// advantage over those with lower scores. This *can* over-select in cases where the fitness
/// landscape is noisy, and a [`RankSelector`] may be more appropriate in those cases.
///
/// This requires the fitness type to be able to be summed up, and then to be divisible by that sum
/// to produce a non-NaN float representing the "portion" of that total fitness the individual in
/// question possesses. The easiest type that implements this is [`NotNan<f64>`].
pub struct RouletteSelector;
impl<I: Individual> ProbabilitySelector<I> for RouletteSelector
where
    I::Fitness: Sum + Div<Output = NotNan<f64>>,
{
    fn probabilities(&self, population: &Population<I>) -> Vec<f64> {
        // Assign each individual a probability of being chosen proportional to their fitness
        let total_fitness = population.individuals().map(|i| i.fitness()).sum();

        population
            .individuals()
            .map(|i| i.fitness() / total_fitness)
            .map(|p| *p)
            .collect()
    }

    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {}
}
