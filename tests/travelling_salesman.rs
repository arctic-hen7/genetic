use genetic::selectors::*;
use genetic::{
    FitnessThresholdStopCriterion, Individual, NullTracker, SimulationBuilder, StaticElitistUpdater,
};
use ordered_float::NotNan;
use rand::{seq::SliceRandom, Rng};

/// The number of cities under consideration.
const NUM_CITIES: usize = 17;
/// Distances between each city (symmetric).
static DISTANCES: [[f64; NUM_CITIES]; NUM_CITIES] = [
    [
        0.0, 633.0, 257.0, 91.0, 412.0, 150.0, 80.0, 134.0, 259.0, 505.0, 353.0, 324.0, 70.0,
        211.0, 268.0, 246.0, 121.0,
    ],
    [
        633.0, 0.0, 390.0, 661.0, 227.0, 488.0, 572.0, 530.0, 555.0, 289.0, 282.0, 638.0, 567.0,
        466.0, 420.0, 745.0, 518.0,
    ],
    [
        257.0, 390.0, 0.0, 228.0, 169.0, 112.0, 196.0, 154.0, 372.0, 262.0, 110.0, 437.0, 191.0,
        74.0, 53.0, 472.0, 142.0,
    ],
    [
        91.0, 661.0, 228.0, 0.0, 383.0, 120.0, 77.0, 105.0, 175.0, 476.0, 324.0, 240.0, 27.0,
        182.0, 239.0, 237.0, 84.0,
    ],
    [
        412.0, 227.0, 169.0, 383.0, 0.0, 267.0, 351.0, 309.0, 338.0, 196.0, 61.0, 421.0, 346.0,
        243.0, 199.0, 528.0, 297.0,
    ],
    [
        150.0, 488.0, 112.0, 120.0, 267.0, 0.0, 63.0, 34.0, 264.0, 360.0, 208.0, 329.0, 83.0,
        105.0, 123.0, 364.0, 35.0,
    ],
    [
        80.0, 572.0, 196.0, 77.0, 351.0, 63.0, 0.0, 29.0, 232.0, 444.0, 292.0, 297.0, 47.0, 150.0,
        207.0, 332.0, 29.0,
    ],
    [
        134.0, 530.0, 154.0, 105.0, 309.0, 34.0, 29.0, 0.0, 249.0, 402.0, 250.0, 314.0, 68.0,
        108.0, 165.0, 349.0, 36.0,
    ],
    [
        259.0, 555.0, 372.0, 175.0, 338.0, 264.0, 232.0, 249.0, 0.0, 495.0, 352.0, 95.0, 189.0,
        326.0, 383.0, 202.0, 236.0,
    ],
    [
        505.0, 289.0, 262.0, 476.0, 196.0, 360.0, 444.0, 402.0, 495.0, 0.0, 154.0, 578.0, 439.0,
        336.0, 240.0, 685.0, 390.0,
    ],
    [
        353.0, 282.0, 110.0, 324.0, 61.0, 208.0, 292.0, 250.0, 352.0, 154.0, 0.0, 435.0, 287.0,
        184.0, 140.0, 542.0, 238.0,
    ],
    [
        324.0, 638.0, 437.0, 240.0, 421.0, 329.0, 297.0, 314.0, 95.0, 578.0, 435.0, 0.0, 254.0,
        391.0, 448.0, 157.0, 301.0,
    ],
    [
        70.0, 567.0, 191.0, 27.0, 346.0, 83.0, 47.0, 68.0, 189.0, 439.0, 287.0, 254.0, 0.0, 145.0,
        202.0, 289.0, 55.0,
    ],
    [
        211.0, 466.0, 74.0, 182.0, 243.0, 105.0, 150.0, 108.0, 326.0, 336.0, 184.0, 391.0, 145.0,
        0.0, 57.0, 426.0, 96.0,
    ],
    [
        268.0, 420.0, 53.0, 239.0, 199.0, 123.0, 207.0, 165.0, 383.0, 240.0, 140.0, 448.0, 202.0,
        57.0, 0.0, 483.0, 153.0,
    ],
    [
        246.0, 745.0, 472.0, 237.0, 528.0, 364.0, 332.0, 349.0, 202.0, 685.0, 542.0, 157.0, 289.0,
        426.0, 483.0, 0.0, 336.0,
    ],
    [
        121.0, 518.0, 142.0, 84.0, 297.0, 35.0, 29.0, 36.0, 236.0, 390.0, 238.0, 301.0, 55.0, 96.0,
        153.0, 336.0, 0.0,
    ],
];

const MUTATION_RATE: f64 = 0.5;

/// A solution to the travelling salesman problem, which is an order in which to visit the cities.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Solution([usize; NUM_CITIES]);
impl Individual for Solution {
    type Fitness = NotNan<f64>; // f64 can only be ordered if NaN disallowed
    type Tracker = NullTracker;

    fn generate(rng: &mut impl Rng) -> Self {
        let mut cities = [0usize; NUM_CITIES];
        for (idx, city) in cities.iter_mut().enumerate() {
            *city = idx;
        }

        cities.shuffle(rng);
        Solution(cities)
    }

    fn tracker() -> Self::Tracker {
        NullTracker
    }

    // Computes the distance of the path using the distances matrix (we aim to minimise this, so
    // then inverted)
    fn fitness(&self) -> Self::Fitness {
        let mut distance = 0.0;
        for i in 0..NUM_CITIES {
            let from = self.0[i];
            let to = self.0[(i + 1) % NUM_CITIES];
            distance += DISTANCES[from][to];
        }

        // Inverting because we maximise fitness by minimising distance
        NotNan::new(1.0 / distance).unwrap()
    }

    // Cross by picking some cities from the first parent, and then filling in the rest from the
    // second parent (both continuous ranges)
    fn cross(&self, other: &Self, _tracker: &Self::Tracker, rng: &mut impl Rng) -> Self {
        let mut child = [0usize; NUM_CITIES];
        let mut used = [false; NUM_CITIES];

        // We allow 0 becuase this generates in the range 0-4 inclusive, so we could feasibly get
        // all first or all second this way (fair to both parents)
        let turn = rng.random_range(0..NUM_CITIES);
        for i in 0..turn {
            child[i] = self.0[i];
            used[self.0[i]] = true;
        }

        // Go through all the cities in the other parent and copy the ones in which we haven't seen
        // yet to the section after the turn (i.e. no longer segment-based, but fill-in-based)
        let mut idx = turn;
        for city in &other.0 {
            if !used[*city] {
                child[idx] = *city;
                idx += 1;
            }
        }

        Solution(child)
    }

    // Mutate by swapping two indices in `MUTATION_RATE * 100`% of cases
    fn mutate(&mut self, _tracker: &Self::Tracker, rng: &mut impl Rng) {
        if rng.random_bool(MUTATION_RATE) {
            let idx1 = rng.random_range(0..NUM_CITIES);
            let idx2 = rng.random_range(0..NUM_CITIES);

            self.0.swap(idx1, idx2);
        }
    }
}

#[test]
fn should_solve_travelling_salesman() {
    const MAX_GENERATIONS: usize = 100;

    let solutions = SimulationBuilder::<Solution>::new()
        .with_initial_population_size(10_000)
        .with_selector(TournamentSelector::new(150))
        .with_updater(StaticElitistUpdater::new_worst(10))
        .with_stop_criterion(FitnessThresholdStopCriterion::new(
            NotNan::new(1.0 / 2085.0).unwrap(),
        ))
        .or_max_generations(MAX_GENERATIONS)
        // .with_seed(9228755800967557821) // 7 generations to 2085 (known global optimum)
        .with_seed(16953019627624396133) // 5 generations to 2085!
        .run()
        .unwrap();

    println!("Best solution: {:?}", *solutions.best().unwrap());
    println!("Fitness: {}", 1.0 / *solutions.best().unwrap().fitness());
    println!("Achieved in {} generations", solutions.num_generations());
    println!("Seed: {}", solutions.seed());
}
