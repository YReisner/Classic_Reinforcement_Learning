# -*- coding: utf-8 -*-

from World import World
import numpy as np
import matplotlib.pyplot as plt

def compare(q_values):
    '''
    Given a Q_values matrix, calculates the sum of squared differences from the DP estimation of the same discount, 0.9
    :param q_values: 16x4 matrix, the q_values given by a Sarsa or Qlearning training.
    :return: Returns the SSE compared to the DP estimation
    '''
    origin = [0,0.217,0.029,-0.032,0.632,0.478,0,-0.117,0.795,0.485,0.129,0.032,0,0,0,-0.117] # values generated in assigment 1
    estimate = np.amax(q_values,axis=1)
    result = np.sum(np.power(np.subtract(origin,estimate),2))
    return result


def plot_tuning_results(results,parameter_values,parameter):
    '''
    Plots the sum of squared differences on the y axis, and the parameter values on the x axis.
    :param results: List of SSE results for each parameter training
    :param parameter_values: List, the values given the the parameter tuned
    :param parameter: String, the name of the parameter tuned
    :return: Does not return anything, only plots the results
    '''
    plt.plot(parameter_values,results)
    plt.title('Sum of Squared differences from DP estimation')
    plt.xlabel(parameter)
    plt.ylabel('values')
    plt.show()


def hyper_prameter_tuning(parameter_values,parameter,method):
    '''
    Performs hyper parameter tuning on a specific hyper-parameter for a list of possible values
    :param parameter_values: A list of values to use for the chosen Parameter
    :param parameter: String, the name of the parameter
    :param method: String, if 'q' then performs hyper parameter with Qlearning method, sarsa otherwise.
    :return: The best hyper-parameter value found, that minimizes the sum of squared differences from the DP estimation
    '''
    results = []
    if method == 'q':
        if parameter == 'alpha':
            for value in parameter_values:
                q, p = env.Qlearning(100000, value, 10)
                difference = compare(q)
                results.append(difference)
            plot_tuning_results(results,parameter_values,parameter)
            best_parameter = parameter_values[np.argmin(results)]
        else:
            for value in parameter_values:
                q, p = env.Qlearning(100000,0.01,value)
                difference = compare(q)
                results.append(difference)
            plot_tuning_results(results,parameter_values,parameter)
            best_parameter = parameter_values[np.argmin(results)]
    else:
        if parameter == 'alpha':
            for value in parameter_values:
                q, p = env.sarsa(100000, value, 10)
                difference = compare(q)
                results.append(difference)
            plot_tuning_results(results,parameter_values,parameter)
            best_parameter = parameter_values[np.argmin(results)]
        else:
            for value in parameter_values:
                q, p = env.sarsa(100000,0.01,value)
                difference = compare(q)
                results.append(difference)
            plot_tuning_results(results,parameter_values,parameter)
            best_parameter = parameter_values[np.argmin(results)]

    print('The best %s is %f' %(parameter,best_parameter))
    return best_parameter






if __name__ == "__main__":
    env = World()
    parameter_tuning = False # change to true in order to perform hyper parameter tuning
    use_sarsa = True # change to true in order to perform choose sarsa as the training method, both with the found optimal values
    if parameter_tuning:
       values = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
       best = hyper_prameter_tuning(values,'alpha','q')
    if use_sarsa:
        q, p = env.sarsa(100000, 0.01, 10)
    else:
        q,p = env.Qlearning(100000,0.05,40)
    env.plot_actionValues(q)
    Pi = np.argmax(p, axis=1) + 1
    env.plot_policy(Pi)

    # Take one final trip with your trained agent :-)
    done = False
    t = 0
    env.reset()
    env.show()
    env.render()
    while not done:
        print("state=", env.observation[0])
        action = env.get_action(env.observation[0],p)
        print("action=",action)
        next_state, reward, done = env.step(action)  # take a random action
        env.render()
        print("next_state",next_state)
        print("env.observation[0]",env.observation[0])
        print("done",done)
        env.observation = [next_state];
        env.close()
        t += 1
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        #input()