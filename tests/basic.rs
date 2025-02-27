//! A collection of basic optimisation tests based on fairly simple continuous optimisation
//! problems.

use genetic::selectors::*;
use genetic::*;

// *Could* spuriously fail once or twice with really bad random numbers
#[test]
fn should_minimise_hyperbowl() {
    const N: usize = 10;
    const MUTATION_PROBABILITY: f64 = 0.2;
    const POPULATION_SIZE: usize = 10_000;

    // The objective function is the sum of the squares of the coordinates
    let objective_fn = |coords: [f64; N]| coords.iter().map(|x| x.powi(2)).sum();

    let solutions = SimulationBuilder::<RealSolution<N>>::new()
        .with_generator(RealSolutionGenerator::new(
            objective_fn,
            true,
            MUTATION_PROBABILITY,
            -1.0..1.0,
            -5000.0..5000.0,
        ))
        .with_initial_population_size(POPULATION_SIZE)
        .with_selector(RouletteSelector.wrap())
        .with_updater(StaticElitistUpdater::new_worst(10))
        .max_generations(300)
        .run()
        .unwrap();

    let best_fitness = 1.0 / *solutions.best().unwrap().fitness();
    println!(
        "Best solution: {:?}",
        solutions.best().unwrap().into_inner().inner
    );
    println!("Fitness: {}", best_fitness);
    println!("Achieved in {} generations", solutions.num_generations());
    println!("Seed: {}", solutions.seed());

    if best_fitness.abs() > 1e-3 {
        panic!("best fitness was {best_fitness}, expected at least 1e-3");
    }
}
