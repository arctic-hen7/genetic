use genetic::selectors::*;
use genetic::*;
use std::f64::consts::PI;
use std::ops::Range;
use std::path::Path;

// *Could* spuriously fail once or twice with really bad random numbers
fn simulate_and_visualise_fn<const N: usize>(
    objective_fn: fn([f64; N]) -> f64,
    gen_range: Range<f64>,
) where
    VisualTracker<RealSolution<N>, N>: Tracker<RealSolution<N>> + Visualise<N>,
{
    let solutions = SimulationBuilder::<RealSolution<N>>::new()
        .with_generator(RealSolutionGenerator::new(
            objective_fn,
            true,
            0.2,
            -1.0..1.0,
            gen_range,
        ))
        .with_initial_population_size(10_000)
        // .with_selector(RouletteSelector.wrap())
        .with_selector(TournamentSelector::new(150))
        .with_updater(StaticElitistUpdater::new_worst(10))
        .with_tracker(VisualTracker::new_min(objective_fn))
        .max_generations(10)
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

    solutions.tracker::<VisualTracker<_, N>, _>().visualise(
        Path::new("visualiser.gif"),
        "Genetic Algorithm Convergence",
        std::array::from_fn(|_| -5.0..5.0),
        0.0..50.0,
        0.01,
        250,
    );

    println!("Evolution GIF produced.")
}

#[test]
fn should_minimise_bowl() {
    let obj_fn = |coords: [f64; 2]| coords.iter().map(|x| x.powi(2)).sum();
    simulate_and_visualise_fn(obj_fn, -5000.0..5000.0);
}

#[test]
fn should_minimise_rastrigin() {
    let obj_fn = |coords: [f64; 2]| {
        20.0 + coords
            .iter()
            .map(|x| x.powi(2) - 10.0 * (2.0 * PI * x).cos())
            .sum::<f64>()
    };

    simulate_and_visualise_fn::<2>(obj_fn, -100.0..100.0)
}

#[test]
fn should_minimise_rosenbrock() {
    let obj_fn = |coords: [f64; 2]| {
        let x = coords[0];
        let y = coords[1];

        // Global min at (2, 4)
        (2.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
    };

    // 18080457605309716425 with 20 gens shows how it moves very well
    simulate_and_visualise_fn::<2>(obj_fn, -5000.0..5000.0);
}
