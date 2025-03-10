import pytest
import numpy as np
from datetime import datetime
from pandas import Timestamp
from epydemix.model.epimodel import EpiModel, stochastic_simulation

# filepath: epydemix/tests/test_epimodel.py

@pytest.fixture
def mock_epimodel():
    model = EpiModel(
        compartments=["Susceptible", "Infected", "Recovered"],
        parameters={"transmission_rate": 0.3, "recovery_rate": 0.1}
    )
    model.add_transition("Susceptible", "Infected", "mediated", ("transmission_rate", "Infected"))
    model.add_transition("Infected", "Recovered", "spontaneous", "recovery_rate")
    return model

def test_stochastic_simulation(mock_epimodel):
    T = 10
    N = 3
    contact_matrices = [{"overall": np.ones((N, N))} for _ in range(T)]
    initial_conditions = np.array([[1000, 0, 0], [0, 10, 0], [0, 0, 0]])
    parameters = {
        "transmission_rate": np.full(T, 0.3),
        "recovery_rate": np.full(T, 0.1)
    }
    dt = 1.0

    compartments_evolution, transitions_evolution = stochastic_simulation(
        T=T,
        contact_matrices=contact_matrices,
        epimodel=mock_epimodel,
        parameters=parameters,
        initial_conditions=initial_conditions,
        dt=dt
    )

    assert compartments_evolution.shape == (T, 3, N)
    assert transitions_evolution.shape == (T, 2, N)

    # Check if the initial conditions are correctly set
    assert np.array_equal(compartments_evolution[0], initial_conditions)

    # Check if the population remains constant
    for t in range(T):
        assert np.sum(compartments_evolution[t]) == np.sum(initial_conditions)

def test_stochastic_simulation_no_transitions(mock_epimodel):
    mock_epimodel.clear_transitions()
    T = 10
    N = 3
    contact_matrices = [{"overall": np.ones((N, N))} for _ in range(T)]
    initial_conditions = np.array([[1000, 0, 0], [0, 10, 0], [0, 0, 0]])
    parameters = {
        "transmission_rate": np.full(T, 0.3),
        "recovery_rate": np.full(T, 0.1)
    }
    dt = 1.0

    with pytest.raises(ValueError, match="The model has no transitions defined. Please add transitions before running simulations."):
        stochastic_simulation(
            T=T,
            contact_matrices=contact_matrices,
            epimodel=mock_epimodel,
            parameters=parameters,
            initial_conditions=initial_conditions,
            dt=dt
        )

def test_stochastic_simulation_invalid_initial_conditions(mock_epimodel):
    T = 10
    N = 3
    contact_matrices = [{"overall": np.ones((N, N))} for _ in range(T)]
    initial_conditions = np.array([[1000, 0], [0, 10], [0, 0]])  # Invalid shape
    parameters = {
        "transmission_rate": np.full(T, 0.3),
        "recovery_rate": np.full(T, 0.1)
    }
    dt = 1.0

    with pytest.raises(ValueError):
        stochastic_simulation(
            T=T,
            contact_matrices=contact_matrices,
            epimodel=mock_epimodel,
            parameters=parameters,
            initial_conditions=initial_conditions,
            dt=dt
        )