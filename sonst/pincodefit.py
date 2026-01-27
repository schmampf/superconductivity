import sys
import pickle
import numpy as np
from pygad import GA as GeneticAlgorithm
from importlib import reload
import hashlib


def hash_genes(genes: np.ndarray) -> str:
    return hashlib.sha1(genes.tobytes()).hexdigest()


sys.path.append("/Users/oliver/Documents/p5control-bluefors-evaluation")

from utilities.basefunctions import bin_y_over_x

reload(sys.modules["utilities.basefunctions"])


def get_Carlos_IV(file="carlosIV.pickle") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Carlos IV data from a pickle file.
    Args:
        file (str): The name of the pickle file containing the data.
    Returns:
        tuple: A tuple containing the current, voltage, and transmission arrays.
    """
    with open(file, "rb") as file:
        theo = pickle.load(file)

    I_theo = np.copy(theo["current"])
    V = np.copy(theo["voltage"])
    tau = np.copy(theo["transmission"])

    return I_theo, V, tau


def update_generation(ga_instance):
    # Increment generation counter stored in the custom GA object
    ga_instance.current_generation += 1


class GeneticAlgorithmPincode(GeneticAlgorithm):
    def __init__(
        self,
        I_exp,
        V_exp,
        theoretical_model: str = "carlosIV",
        max_number_of_channels: int = 10,
        number_of_individuals: int = 100,
        number_of_generations: int = 1000,
    ):

        self.I_theo, self.V_theo, self.tau_theo = self.load_theoretical_model(
            theoretical_model
        )

        super().__init__(
            sol_per_pop=number_of_individuals + 1,
            num_parents_mating=int(number_of_individuals + 1 / 2),
            num_generations=number_of_generations,
            fitness_func=self.get_fitness,
            num_genes=max_number_of_channels,
            gene_type=int,
            init_range_low=0,
            init_range_high=len(self.tau_theo),
            gene_space=list(range(len(self.tau_theo))),
            mutation_percent_genes="default",
            mutation_type="random",
            crossover_type="single_point",
            on_generation=update_generation,
        )

        self.I_exp = I_exp

        self.current_generation = 0

        self.SUMMARY = []

    def load_theoretical_model(
        self, theoretical_model: str = "carlosIV"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if theoretical_model == "carlosIV":
            I_theo, V_theo, tau_theo = get_Carlos_IV(file="carlosIV.pickle")
        else:
            raise ValueError(f"Unknown theoretical model: {theoretical_model}")

        return I_theo, V_theo, tau_theo

    def get_i_theo_from_genes(self, genes: np.ndarray) -> np.ndarray:
        """
        genes: 1D array of integers [0 up to len(transmission)] with length max_number_of_channels
        """
        return np.sum(self.I_theo[genes, :], axis=0)

    def error_function(self, i_theo: np.ndarray) -> float:
        return np.nansum((i_theo - self.I_exp) ** 2)

    def get_fitness(self, ga, genes: np.ndarray, index_of_individum) -> float:
        i_theo = self.get_i_theo_from_genes(genes)
        fitness = -self.error_function(i_theo)
        self.SUMMARY.append(
            {
                "generation": self.current_generation,
                "index": index_of_individum,
                "genes": np.copy(genes),
                "fitness": fitness,
                # "id": hash_genes(genes),
            }
        )
        return fitness

    def run_get_pincode(
        self,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self.run()
        solution, fitness, _ = self.best_solution()
        solution = np.sort(solution)
        tau_fit = self.tau_theo[solution]
        I_fit = self.get_i_theo_from_genes(solution)
        error_fit = float(fitness)
        return I_fit, tau_fit, error_fit


def get_pincode(
    I_exp: np.ndarray,
    V_exp: np.ndarray,
    theoretical_model: str = "carlosIV",
    max_number_of_channels: int = 10,
    number_of_individuals: int = 100,
    number_of_generations: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Get the pincode for the given experimental data.
    """
    GA = GeneticAlgorithmPincode(
        I_exp,
        V_exp,
        theoretical_model,
        max_number_of_channels=max_number_of_channels,
        number_of_individuals=number_of_individuals,
        number_of_generations=number_of_generations,
    )
    V_theo = GA.V_theo
    I_fit, tau_fit, error_fit = GA.run_get_pincode()

    return I_fit, V_theo, tau_fit, error_fit


def get_printcode(
    I_exp: np.ndarray,
    V_exp: np.ndarray,
    theoretical_model: str = "carlosIV",
    max_number_of_channels: int = 10,
    number_of_individuals: int = 100,
    number_of_generations: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list]:
    """
    Get the pincode for the given experimental data.
    """
    GA = GeneticAlgorithmPincode(
        I_exp,
        V_exp,
        theoretical_model,
        max_number_of_channels=max_number_of_channels,
        number_of_individuals=number_of_individuals,
        number_of_generations=number_of_generations,
    )
    V_theo = GA.V_theo
    I_fit, tau_fit, error_fit = GA.run_get_pincode()

    return I_fit, V_theo, tau_fit, error_fit, GA.SUMMARY
