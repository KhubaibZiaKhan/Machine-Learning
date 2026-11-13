from numpy import asarray, arange
from numpy.random import rand
from matplotlib import pyplot

def objective (x):
    return x**2.0


def derivative (x):
    return x * 2.0

def gradient_descent(objective, derivative, bounds, n_iter, step_size):
    solutions, scores = list(), list()

    solution = bounds[:, 0] + rand(len(bounds)) *  (bounds[:,1] - bounds[:, 0])

    for i in range(n_iter):

        gradient = derivative(solution)

        solution = solution - step_size * gradient

        solution_eval = objective(solution)

        solutions.append(solution.copy())
        scores.append(solution_eval.copy())

        print('>%d f(%s) = %.5f' % (i, solution[0], solution_eval[0]))
    return [solution, scores]

bounds = asarray([[-1.0,1.0]])
n_iter = 30
step_size = 0.1

solutions, scores = gradient_descent(
    objective, derivative, bounds, n_iter, step_size
)

inputs = arange(bounds[0, 0], bounds[0, 1] + 0.1, 0.1)

results = objective(inputs)

pyplot.plot(inputs, results, label='objective function')
pyplot.plot(
    asarray(solutions).flatten(),
    asarray(scores).flatten(),
    '.-',
    color='red',
    label='Gradient Descent'
)
pyplot.xlabel('Input (x)')
pyplot.ylabel('Objective Function f(x)')
pyplot.title('Gradient Decent Optimization')
pyplot.legend()
pyplot.grid(True)
pyplot.show()