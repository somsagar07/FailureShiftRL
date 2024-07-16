import math

from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

import numpy as np
import plotly.graph_objects as go
import torch


# Entropy:
# This function can be used to calculate entropy of the system. 
# You need to pass the episode actions and the correct actions and it will return the entropy of the system.
def calculate_entropy(episode_actions, correct_actions):
    correct_counts = {}
    for action, correct in zip(episode_actions, correct_actions):
        if correct == 1:
            if action in correct_counts:
                correct_counts[action] += 1
            else:
                correct_counts[action] = 1

    sorted_actions = sorted(correct_counts.keys())
    sorted_counts = [correct_counts[action] for action in sorted_actions]

    d = {}
    for action, count in zip(sorted_actions, sorted_counts):
        d[action] = count
    
    total = sum(d.values())
    entropy = 0
    for key in d:
        p = d[key] / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# Wasserstein Distance:
# You have to pass the pretrained proxy mean and fine-tuned proxy mean to the function and it will calculate the wasserstein 
# distance between the two distributions. You can use the values we got (defined below) to test the function.
def w_distance(distribution1, distribution2):
    wasserstein_distance_value = wasserstein_distance(distribution1, distribution2)
    print(f"Wasserstein distance between the distributions: {wasserstein_distance_value}")
    return wasserstein_distance_value


# Probability Shift: Continuous and Discrete
# These functions were used to calculate action probability shifts in the RL models. For continous action space you can use 
# the continous_density_shift function while for discrete action space you can use the discrete_density_shift function. You 
# only have to pass the actions taken by the agent with pretrained model and after finetuning and retraining the rl model what
# were new actions taken during the same rollouts. The function will return the plots of action probability shift for each action.
def continous_density_shift(pretrained_actions, finetuned_actions):
    # Assuming list1 and list2 are your data lists
    list1 = np.array(pretrained_actions)  # Make sure these are numpy arrays
    list2 = np.array(finetuned_actions)

    # Reshaping data for KDE
    list1 = list1.reshape(-1, 1)
    list2 = list2.reshape(-1, 1)

    # Setting up KDE for both datasets
    kde1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(list1)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(list2)

    # Creating an array of values to evaluate the KDE on
    x_d = np.linspace(min(np.min(list1), np.min(list2)), max(np.max(list1), np.max(list2)), 1000)
    x_d = x_d.reshape(-1, 1)

    # Evaluating KDE for both sets
    log_dens1 = kde1.score_samples(x_d)
    log_dens2 = kde2.score_samples(x_d)

    # Plotting
    plt.fill_between(x_d[:, 0], np.exp(log_dens1), alpha=0.5, label='Pretrained')
    plt.fill_between(x_d[:, 0], np.exp(log_dens2), alpha=0.5, label='Finetuned')

    # Adding labels and title
    plt.xlabel('Action', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title('Action Probability Shift', fontsize=14)

    # Create legend & Show graphic
    plt.legend(fontsize=11)
    plt.show()

def discrete_density_shift(pretrained_actions, finetuned_actions):
    unique_actions_pre, action_counts_pre = np.unique(pretrained_actions, return_counts=True)
    unique_actions_post, action_counts_post = np.unique(finetuned_actions, return_counts=True)

    width = 0.4
    space = 0

    unique_actions_pre_adjusted = unique_actions_pre - (width + space) / 2
    unique_actions_post_adjusted = unique_actions_post + (width + space) / 2

    plt.bar(unique_actions_pre_adjusted, action_counts_pre, width, label='Pretrained')
    plt.bar(unique_actions_post_adjusted, action_counts_post, width, label='Finetuned')
    plt.xlabel('Action', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    plt.title('Discrete Actions Shift', fontsize=16)
    plt.show()


# Barry Center:
# This is the function to find barry center of the 3d heat map. You need to provide the following parameters and the function 
# will plot the barry center on that 3d heat map with the given parameters. Parameters are:
# 1. means: This is the proxy mean for the model given by the agent.
# 2. calculate_scale: This is the scale calculated based on rank calculated based on log of standard deviation.
# 3. center: This is the center of the sphere in which you want to find the barry center.
# 4. radius: This is the radius of the sphere in which you want to find the barry center.
# 5. threshold: This is the threshold for the certainty of the barry center.
# 6. p_std: This is the proxy standard deviation for the model given by the agent.
def find_rank_of_number(num_list, number):
    new_list = num_list.copy()
    new_list.append(number)
    new_list.sort()
    rank = new_list.index(number) + 1  
    return rank

def calculate_scale(proxy_mean, proxy_std):
    proxy_std_log = 1*torch.log(proxy_std).cpu()
    proxy_std_log_np = proxy_std_log.detach().numpy()
    sorted_indices = np.argsort(proxy_std_log_np)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(proxy_std_log_np))
    ranks += 1 
    ranks_tensor = torch.tensor(ranks).cpu()
    scale = (ranks_tensor.reshape(5, 5, 5).detach().cpu().numpy())/2

    return scale

def plot_with_barycenter_stats(means, scale, center, radius, threshold, p_std):
    # Generate grid indices
    xs, ys, zs = np.indices(means.shape)

    # Flatten the arrays for easier processing
    xs_flat, ys_flat, zs_flat = xs.flatten(), ys.flatten(), zs.flatten()
    means_flat, stds_flat = means.flatten(), scale.flatten()

    # Compute mask for points within the specified radius
    within_radius_mask = np.sqrt((xs_flat - center[0])**2 + (ys_flat - center[1])**2 + (zs_flat - center[2])**2) <= radius
    outside_radius_mask = ~within_radius_mask

    # Compute barycenter of points within the radius
    barycenter_x = np.mean(xs_flat[within_radius_mask])
    barycenter_y = np.mean(ys_flat[within_radius_mask])
    barycenter_z = np.mean(zs_flat[within_radius_mask])

    # Compute mean and standard deviation for the barycenter
    barycenter_mean = np.mean(means_flat[within_radius_mask])
    
    num_points = np.sum(within_radius_mask)
   
    b_std = p_std.cpu().detach().numpy()
    barycenter_std = np.sum((means.flatten()[within_radius_mask] - barycenter_mean)**2 + b_std[within_radius_mask]**2)/num_points
    barycenter_std = np.sqrt(barycenter_std)
    
    
    barycenter_std = np.log(barycenter_std)
   
    barycenter_std = find_rank_of_number(list(np.log(b_std)), barycenter_std)/2

    # Determine certain and uncertain points outside the radius
    certain_mask = (means_flat > threshold) & outside_radius_mask
    uncertain_mask = (means_flat <= threshold) & outside_radius_mask

    # Trace for certain points outside the radius
    trace1 = go.Scatter3d(
        x=xs_flat[certain_mask],
        y=ys_flat[certain_mask],
        z=zs_flat[certain_mask],
        mode='markers',
        name='Certain Area',
        marker=dict(
            size=stds_flat[certain_mask],
            symbol='circle',
            color=np.log(means_flat[certain_mask]),
            colorbar=dict(thickness=10, ticklen=4, x=1),
            colorscale='Viridis',
            opacity=0.8,
            line=dict(color='Black', width=1)
        ),
        hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>LogMean: %{marker.color}<extra></extra>",
    )

    # Trace for uncertain points outside the radius
    trace2 = go.Scatter3d(
        x=xs_flat[uncertain_mask],
        y=ys_flat[uncertain_mask],
        z=zs_flat[uncertain_mask],
        mode='markers',
        name='Uncertain Area',
        marker=dict(
            size=10,
            symbol='square',
            color='rgb(255,0,0)',
            opacity=0.8,
            line=dict(color='Black', width=1)
        ),
    )

    # Trace for barycenter
    trace3 = go.Scatter3d(
        x=[barycenter_x],
        y=[barycenter_y],
        z=[barycenter_z],
        mode='markers',
        name='Barycenter',
        marker=dict(
            size=barycenter_std/2,  # Scale size by standard deviation
            symbol='diamond',
            color=np.log(barycenter_mean),  # Color scaled by the logarithm of the mean
            colorscale='Viridis',
            opacity=1,
            line=dict(color='Black', width=2)
        ),
        hovertemplate=f"Barycenter<br>X: {barycenter_x:.2f}<br>Y: {barycenter_y:.2f}<br>Z: {barycenter_z:.2f}<br>LogMean: {np.log(barycenter_mean):.2f}<extra></extra>",
    )

    layout = go.Layout(
        height=800,
        width=800,
        title='3D Heatmap with Barycenter Stats',
        scene=dict(
            xaxis=dict(title='Rotation'),
            yaxis=dict(title='Darken'),
            zaxis=dict(title='Saturation')
        ),
    )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.show()