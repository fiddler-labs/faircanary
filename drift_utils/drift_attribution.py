import numpy as np
import torch
from sklearn.utils.extmath import cartesian
from copy import deepcopy
from captum.attr import LayerIntegratedGradients, IntegratedGradients
import logging

STEPS = 20
MAX_STEPS = 5000
EPS = 1e-8


def drift_shapley(reference, target, drift_score_fn):
    """

    :param reference: the reference (baseline) dataset of shape (M, N) for the
                      drift calculation. M is the number of rows or point
                      instances that serve as input to the model. N is the
                      number of features
    :param target: the dataset of shape (M,N) whose drift we want to calculate,
                    as compared to the reference dataset
    :param drift_score_fn: the value function for the Shapley value
                           calculation, a combination of the model function
                           and a function or metric that calculates
                           distributional distance
    :return:
        attribution vector of shape N, consisting of exact Shapley value
        contributions of each feature to the overall prediction drift
    """
    m, n = reference.shape
    assert m, n == target.shape
    permutation_mask = cartesian([[False, True]] * n)
    p, _ = permutation_mask.shape
    permutation_mask_reshaped = permutation_mask.reshape(p, 1, n)
    permuted_data = np.repeat(target.reshape(1, -1), [p],
                              axis=0).ravel().reshape(p, m,
                                                      n) * permutation_mask_reshaped + \
                    np.repeat(reference.reshape(1, -1), [p],
                              axis=0).ravel().reshape(p, m, n) * (
                        ~permutation_mask_reshaped)

    # print('permuted data', permuted_data)
    drift_scores = drift_score_fn(reference, permuted_data)
    # print(f'drift scores are {drift_scores}')
    attrs = [0] * n
    n_fact = np.math.factorial(n)
    for i in range(n):
        non_zero_indices = (permutation_mask[:, i] == True).nonzero()[
            0].tolist()
        #print(f'Non zero indices are {non_zero_indices}')

        for non_zero_index in non_zero_indices:
           #print(f'Non zero index {non_zero_index}')
            m_row = deepcopy(permutation_mask[non_zero_index, :])
            m_row[i] = False
            complement_row_idx = \
                np.where((permutation_mask == m_row).all(axis=1))[0].tolist()[
                    0]
            complement_row_sum = np.sum(m_row)
            if complement_row_sum == 0:
                #print(f'i is {i}')
                attrs[i] += np.math.factorial(n - 1) * (
                    drift_scores[non_zero_index]) / n_fact
                #print(non_zero_index, attrs[i])
            else:
                attrs[i] += np.math.factorial(complement_row_sum) * (
                    drift_scores[non_zero_index] - drift_scores[
                    complement_row_idx]) / n_fact
                #print(non_zero_index, attrs[i])

   #print(attrs, drift_scores)
    assert np.isclose(np.sum(attrs), drift_scores[-1])
    return attrs


def drift_shapley_approximation(reference, target, drift_score_fn,
                                max_number_permutations=None, seed=0):
    """

    :param reference: the reference (baseline) dataset of shape (M, N) for the
                      drift calculation. M is the number of rows or point
                      instances that serve as input to the model. N is the
                      number of features
    :param target: the dataset of shape (M,N) whose drift we want to calculate,
                    as compared to the reference dataset
    :param drift_score_fn: the value function for the Shapley value
                           calculation, a combination of the model function
                           and a function or metric that calculates
                           distributional distance
    :param max_number_permutations: the number of permutations to be
                                     evaluated for the approximation of
                                     Shapley values. If None, the exact
                                     Shapley value will be calculated. The
                                     max number of permutations for N
                                     features is 2^N
    :param seed: the seed for choosing the random permutations
    :return:
    attribution vector of shape N, consisting of approximated Shapley value
        contributions of each feature to the overall prediction drift
    """
    np.random.seed(seed)

    sampling_flag = False
    m, n = reference.shape
    assert m, n == target.shape

    permutation_mask, permuted_data, sampling_flag = \
        _get_shapley_permuted_data_mask_flag(reference, target,
                                             max_number_permutations=max_number_permutations)

    drift_scores = drift_score_fn(reference=reference, target=permuted_data)
    attrs = [0] * n
    divisors = []

    # 1. We loop through every feature
    # 2. We find every index in the permutation mask where this feature is
    # present in the coalition
    # 3. We loop through these indices
    # 4. For each index, we find the complement row where the feature is
    # missing from the coalition
    # 5. We then calculate the attribution of adding the feature to the
    # coalition, by multiplying the value function difference with the
    # appropriate multiplier
    for i in range(n):
        non_zero_indices = (permutation_mask[:, i] == True).nonzero()[
            0].tolist()
        divisor = 0
        for non_zero_index in non_zero_indices:
            m_row = deepcopy(permutation_mask[non_zero_index, :])
            m_row[i] = False
            complement_row_idx = \
                np.where((permutation_mask == m_row).all(axis=1))[0].tolist()
            if len(complement_row_idx) > 0:
                complement_row_idx = complement_row_idx[0]
                complement_row_sum = np.sum(m_row)

                multiplier = _get_shapley_multiplier(complement_row_sum, n)
                divisor += multiplier
                attrs[i] += multiplier * (
                    drift_scores[non_zero_index] - drift_scores[
                    complement_row_idx])
        divisors.append(divisor)
    # print(drift_scores)
    attrs = [attrs[i] / divisors[i] for i in range(n)]
    # if sampling_flag is False:
    #     assert np.isclose(np.sum(attrs), drift_scores[-1])
    return attrs


def drift_shapley_group_approximation(reference, target, drift_score_fn,
                                      group_sizes=None,
                                      max_number_permutations=None, seed=0):
    """
    :param reference: the reference (baseline) dataset of shape (M, N) for the
            drift calculation. M is the number of rows or point
            instances that serve as input to the model. N is the
            number of features

    :param target: the dataset of shape (M,N) whose drift we want to calculate,
            as compared to the reference dataset

    :param drift_score_fn: the value function for the Shapley value
                            calculation, a combination of the model function
                            and a function or metric that calculates
                            distributional distance

    :param group_sizes: a list of group sizes i.e. num of points in each
                    group, starting in order. It is assumed that the
                    reference and target data are aligned by group.

    :param max_number_permutations: the number of permutations to be
                                    evaluated for the approximation of
                                    Shapley values. If None, the exact Shapley
                                    value will be calculated. The max number of
                                     permutations for N features is 2^N
    :param seed: the seed for choosing the random permutations

    :return:
        attribution vector of shape N, consisting of the approximate Shapley
        value contributions of each feature to the overall prediction drift

    """

    np.random.seed(seed)

    sampling_flag = False
    m, n = reference.shape
    assert m, n == target.shape

    if group_sizes is not None:
        num_groups = len(group_sizes)
    else:
        num_groups = 1
    total_factors = num_groups * n  # num of 'features' is now a combination of
    # groups and features
    permutation_mask, sampling_flag = _get_permutation_mask_flag(total_factors,
                                                                 max_number_permutations)
    p_orig, _ = permutation_mask.shape
    drift_scores = [-1] * p_orig
    attrs = np.zeros((num_groups, n))
    divisors = np.zeros((num_groups, n))
    p, _ = permutation_mask.shape
    # factor map:
    # Go by groups and for each group, go by features
    for i in range(total_factors):
        group_id_orig = i // n
        feature_id_orig = i % n
        # This creates an array of shape (p, m, total_factors) of the target
        # data, with the columns where the feature is missing replaced with
        # the corresponding values from the reference dataset
        non_zero_indices = (permutation_mask[:, i] == True).nonzero()[
            0].tolist()
        divisor = 0
        for non_zero_index in non_zero_indices:
            m_row_orig = deepcopy(permutation_mask[non_zero_index, :])
            m_row_complement = deepcopy(m_row_orig)
            m_row_complement[i] = False
            complement_row_idx = \
                np.where((permutation_mask == m_row_complement).all(axis=1))[
                    0].tolist()
            if len(complement_row_idx) > 0:
                complement_row_idx = complement_row_idx[0]
                complement_row_sum = np.sum(m_row_complement)

                multiplier = _get_shapley_multiplier(complement_row_sum,
                                                     total_factors)
                divisor += multiplier

                if drift_scores[non_zero_index] == -1:
                    drift_scores[non_zero_index] = \
                        drift_score_fn(_get_shapley_group_permuted_data(target,
                                                                        reference,
                                                                        group_sizes,
                                                                        m_row_orig),
                                       reference)

                if drift_scores[complement_row_idx] == -1:
                    drift_scores[complement_row_idx] \
                        = drift_score_fn(_get_shapley_group_permuted_data(target,
                                                                          reference,
                                                                          group_sizes,
                                                                          m_row_complement),
                                         reference)
                attrs[group_id_orig][feature_id_orig] += multiplier * (
                    drift_scores[non_zero_index]
                    - drift_scores[complement_row_idx])
        divisors[group_id_orig][feature_id_orig] = divisor

    # print(drift_scores)
    attrs = attrs / divisors
    # if sampling_flag is False:
    #     assert np.isclose(np.sum(attrs), drift_scores[-1])
    return attrs


def drift_layer_integrated_gradients_captum(drift_score_function,
                                            model_layer_to_attribute,
                                            target_index,
                                            reference,
                                            target,
                                            attribute_to_layer_input=False,
                                            auxiliary_inputs=None,
                                            max_allowed_error=1,
                                            steps=20,
                                            batch_size=None
                                            ):
    """
    :param drift_score_function: the value function for the Shapley value
                           calculation, a combination of the model function
                           and a function or metric that calculates
                           distributional distance

    :param model_layer_to_attribute:

    :param target_index: the index of the output vector for which to compute
                         Integrated Gradients

    :param reference: the reference (baseline) dataset of shape (M, N) for the
            drift calculation. M is the number of rows or point
            instances that serve as input to the model. N is the
            number of features

    :param target: the dataset of shape (M,N) whose drift we want to calculate,
            as compared to the reference dataset

    :param attribute_to_layer_input: boolean which determines if
                                     attributions are to be calculated wrt
                                     layer input
    :param auxiliary_inputs: the additional, non-differentiable inputs to
                             the model
    :param max_allowed_error: the maximum error tolerance in the calculation
                              of Integrated Gradients using the efficiency
                              axiom
    :param steps: num of steps for the calculation of the integral
    :param batch_size: the batch size
    :return: the Integrated Gradients attributions for each input and feature

    """

    lig = LayerIntegratedGradients(drift_score_function,
                                   model_layer_to_attribute)
    baseline_prediction = 0
    prediction = drift_score_function(target)
    print(f'The drift prediction is {prediction}')
    percent_error = 100
    while abs(percent_error) > max_allowed_error:
        attributions, delta = lig.attribute(
            target,
            baselines=reference,
            additional_forward_args=auxiliary_inputs,
            target=target_index,
            n_steps=steps,
            internal_batch_size=batch_size,
            return_convergence_delta=True,
            attribute_to_layer_input=attribute_to_layer_input
        )
        # print(f'delta is {delta}')
        if model_layer_to_attribute:
            # if True, then attributions are returned as a tuple
            attributions = attributions[0]
        logging.info(f'attributions shape is {attributions.shape}')
        percent_error = _get_ig_error(torch.sum(attributions),
                                      baseline_prediction,
                                      prediction)
        print(f'Percent error is {percent_error}')
        steps += STEPS
        if steps > MAX_STEPS:
            break

    return attributions


def drift_integrated_gradients_captum(drift_score_function,
                                      target_index,
                                      reference,
                                      target,
                                      auxiliary_inputs=None,
                                      max_allowed_error=1,
                                      steps=20,
                                      batch_size=None
                                      ):
    """

    :param drift_score_function: the value function for the Integrated
                           Gradients calculation, a combination of the model
                           function and a function or metric that calculates
                           distributional distance
    :param target_index: the index of the output vector for which to compute
                         Integrated Gradients
    :param reference: the reference (baseline) dataset of shape (M, N) for the
            drift calculation. M is the number of rows or point
            instances that serve as input to the model. N is the
            number of features

    :param target: the dataset of shape (M,N) whose drift we want to calculate,
            as compared to the reference dataset
    :param auxiliary_inputs: the additional, non-differentiable inputs to
                             the model

    :param max_allowed_error: the maximum error tolerance in the calculation
                              of Integrated Gradients using the efficiency
                              axiom
    :param steps: num of steps for the calculation of the integral
    :param batch_size: the batch size
    :return: the Integrated Gradients attributions for each input and feature
    """

    ig = IntegratedGradients(drift_score_function)
    baseline_prediction = 0
    prediction = drift_score_function(target)
    print(f'The drift prediction is {prediction}')
    percent_error = 100
    while abs(percent_error) > max_allowed_error:
        attributions, delta = ig.attribute(
            target,
            baselines=reference,
            additional_forward_args=auxiliary_inputs,
            target=target_index,
            n_steps=steps,
            internal_batch_size=batch_size,
            return_convergence_delta=True,
        )
        #print(f'delta is {delta}')
        logging.info(f'attributions shape is {attributions.shape}')
        percent_error = _get_ig_error(torch.sum(attributions),
                                      baseline_prediction,
                                      prediction)
        print(f'Percent error is {percent_error}')
        steps += STEPS
        if steps > MAX_STEPS:
            break

    return attributions


def _get_ig_error(attr_sum, baseline_prediction, prediction):
    delta_prediction = prediction - baseline_prediction
    logging.info(f'attr_sum is {attr_sum}')
    error_percentage = 100 * (delta_prediction - attr_sum) / delta_prediction
    logging.info(f'Error percentage is {error_percentage}')
    return error_percentage


def _get_shapley_multiplier(complement_row_sum, n):
    return np.math.factorial(complement_row_sum) * \
           np.math.factorial(n - 1 - complement_row_sum)


def _calc_shapley_attributions(permutation_mask, drift_scores, n,
                               sampling_flag=False):
    attrs = [0] * n
    divisors = []
    for i in range(n):
        non_zero_indices = (permutation_mask[:, i] == True).nonzero()[
            0].tolist()
        # print(f'Non zero indices are {non_zero_indices}')
        divisor = 0
        for non_zero_index in non_zero_indices:
            # print(f'Non zero index {non_zero_index}')
            m_row = deepcopy(permutation_mask[non_zero_index, :])
            m_row[i] = False
            # print(f'new row is  {m_row}')
            complement_row_idx = \
                np.where((permutation_mask == m_row).all(axis=1))[0].tolist()
            # print(f'complement row index is {complement_row_idx}')
            if len(complement_row_idx) > 0:
                complement_row_idx = complement_row_idx[0]
                complement_row_sum = np.sum(m_row)
                # print(f'complement row sum is {complement_row_sum}')

                multiplier = _get_shapley_multiplier(complement_row_sum, n)
                # print('multiplier is', multiplier)
                divisor += multiplier
                attrs[i] += multiplier * (
                    drift_scores[non_zero_index] - drift_scores[
                    complement_row_idx])
        divisors.append(divisor)
        # print(f'for i {i} {attrs[i]}, {divisors[i]}')

    # print(attrs, divisors)
    attrs = [attrs[i] / divisors[i] for i in range(n)]
    # if sampling_flag is False:
    #     #print(drift_scores[-1], np.sum(attrs))
    #     # assert np.isclose(np.sum(attrs), drift_scores[-1])
    return attrs


def _get_shapley_permuted_data_mask_flag(reference, target, group_boundaries=None,
                                         max_number_permutations=None):
    sampling_flag = False  # if false, all possible permutations will be
    # returned

    m, n = reference.shape
    assert m, n == target.shape

    if group_boundaries is None:
        group_boundaries = [0, m]

    start, end = group_boundaries
    group_len = end - start

    #print(f'Group len is {group_len}')

    slice = np.s_[start: end]
    target_slice = target[slice]
    #print(f'Target slice shape is {target_slice.shape}')
    reference_slice = reference[slice]

    target_remainder = np.delete(target, slice, axis=0)
    #print(f'Target remainder shape is {target_remainder.shape}')

    permutation_mask, sampling_flag = _get_permutation_mask_flag(n,
                                                                 max_number_permutations)

    p, _ = permutation_mask.shape

    #print(f'Permutation mask shape is {permutation_mask.shape}')

    permutation_mask_reshaped = permutation_mask.reshape(p, 1, n)
    #print(f'Permutation mask shape is {permutation_mask.shape}')

    permuted_data = np.repeat(target_slice.reshape(1, -1), [p],
                              axis=0).ravel().reshape(p, group_len,
                                                      n) * permutation_mask_reshaped + \
                    np.repeat(reference_slice.reshape(1, -1), [p],
                              axis=0).ravel().reshape(p, group_len, n) * (
                        ~permutation_mask_reshaped)

    if group_len < m:
        non_permuted_data = np.repeat(target_remainder.reshape(1, -1), [p],
                                      axis=0).ravel().reshape(p,
                                                              m - group_len,
                                                              n)
        permuted_data = np.concatenate((permuted_data, non_permuted_data),
                                       axis=1)

    return permutation_mask, permuted_data, sampling_flag


def _get_permutation_mask_flag(n, max_number_permutations):
    permutation_mask = cartesian([[False, True]] * n)
    # print(f'Permutation mask shape is {permutation_mask.shape}')
    sampling_flag = False
    if max_number_permutations is not None:
        if max_number_permutations < len(permutation_mask):
            sampling_fraction = max_number_permutations / len(permutation_mask)
            mask = np.random.choice([True, False], len(permutation_mask),
                                    p=[sampling_fraction,
                                       1 - sampling_fraction])
            permutation_mask = permutation_mask[mask]
            sampling_flag = True

    return permutation_mask, sampling_flag


def _get_shapley_group_permuted_data(data, baseline_data, group_sizes,
                                     permutation_mask_row):
    new_data = deepcopy(data)
    m, n = new_data.shape

    if group_sizes is None:
        return data * permutation_mask_row + baseline_data * (
            ~permutation_mask_row)

    num_groups = len(group_sizes)
    num_factors = n * num_groups
    assert num_factors == len(permutation_mask_row)

    for factor, boolean in enumerate(permutation_mask_row):
        if boolean == 1:
            continue
        else:
            group_id = factor // n
            feature_id = factor % n
            row_start = sum(group_sizes[:group_id])
            row_end = row_start + group_sizes[group_id]
            new_data[row_start: row_end, feature_id] = baseline_data[
                                                       row_start:row_end,
                                                       feature_id]
    return new_data


def _generate_scaled_inputs(reference, target, num_steps=20):
    return [torch.tensor(reference + (i / num_steps) * (target - reference),
                         requires_grad=True) for i in range(num_steps + 1)]


def _calculate_integral(gradients, axis=0):
    riemann_sum = (gradients[:-1] + gradients[1:]) / 2.0  # trapezoidal rule
    integral = torch.mean(riemann_sum, axis=axis)
    return integral
