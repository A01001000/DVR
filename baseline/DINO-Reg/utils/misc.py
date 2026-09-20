import numpy as np

def custom_enumerate(sequence, start=0, step=1):
    n = len(sequence)
    # Ensure the last index is included by adjusting the range if necessary
    indices = range(start, n, step)
    if n - 1 not in indices:
        indices = list(indices) + [n - 1]

    for i in indices:
        yield i, sequence[i]


def calculate_total_difference_for_translation(bestMatch, translation, fixed_set_size):
    # Apply translation to each index in the bestMatch array
    translated_indices = bestMatch + translation

    # Ensure translated indices are within bounds
    translated_indices = np.clip(translated_indices, 0, fixed_set_size - 1)

    # Calculate differences between translated indices and their original indices
    differences = np.abs(np.arange(len(bestMatch)) - translated_indices)

    # Sum the differences
    total_difference = np.sum(differences)

    return total_difference

def find_optimal_translation_within_range(bestMatch, search_range, fixed_set_size):
            optimal_translation = None
            lowest_total_difference = np.inf
            total_differences = np.zeros(len(search_range))

            for idx_trans,translation in enumerate(search_range):
                total_difference = calculate_total_difference_for_translation(bestMatch, translation, fixed_set_size)
                total_differences[idx_trans] = total_difference

                if total_difference < lowest_total_difference:
                    lowest_total_difference = total_difference
                    optimal_translation = translation

            return optimal_translation, total_differences




def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def cosine_similarity_batch(set1, set2):
    # Normalize set1 and set2 along the feature axis
    norm_set1 = set1 / np.linalg.norm(set1, axis=3, keepdims=True)
    norm_set2 = set2 / np.linalg.norm(set2, axis=3, keepdims=True)

    # Calculate the dot products between all pairs of normalized feature vectors
    dot_products = np.sum(norm_set1[:, np.newaxis] * norm_set2, axis=3)

    # Average the dot products over the H and W dimensions to get the cosine similarity
    cosine_similarity = np.mean(dot_products, axis=(2, 3))

    return cosine_similarity

def calculate_best_match_cls(set1, set2, feat_sliceNum, mode='l2'):

            """distance between each moving feature to all fixed features"""
            dist = np.zeros((feat_sliceNum, feat_sliceNum))
            for i in range(feat_sliceNum):
                if mode == 'l2':
                    dist[i, :] = np.linalg.norm(set1[i, :] - set2, axis=1)
                elif mode == 'l2_align':
                    centroid_set1 = np.mean(set1, axis=0)
                    centroid_set2 = np.mean(set2, axis=0)
                    # Calculate the translation needed
                    translation = centroid_set1 - centroid_set2
                    # Apply the translation to pca_set2 to align it with pca_set1
                    aligned_set2 = set2 + translation
                    dist[i, :] = np.linalg.norm(set1[i, :] - aligned_set2, axis=1)

                elif mode == 'cosine':
                    dist[i, :] = [cosine_similarity(set1[i, :], set2[j, :]) for j in range(feat_sliceNum)]
                elif mode == 'cosine_align':
                    centroid_set1 = np.mean(set1, axis=0)
                    centroid_set2 = np.mean(set2, axis=0)
                    # Calculate the translation needed
                    translation = centroid_set1 - centroid_set2
                    # Apply the translation to pca_set2 to align it with pca_set1
                    aligned_set2 = set2 + translation
                    dist[i, :] = [cosine_similarity(set1[i, :], aligned_set2[j, :]) for j in range(feat_sliceNum)]

            """find the closest fixed feature for each moving feature"""
            if mode == 'l2' or mode == 'l2_align':
                closest_feature = np.argmin(dist, axis=1)
            else:
                closest_feature = np.argmax(dist, axis=1)
            return closest_feature


def calculate_best_match_patchT(set1, set2, mode='l2'):

    """move dimensions so that the slice number is the first dimension"""
    set1 = np.moveaxis(set1, 2, 0)
    set2 = np.moveaxis(set2, 2, 0)

    """distance between each moving feature to all fixed features"""
    feat_sliceNum = set1.shape[0]
    dist = np.zeros((feat_sliceNum, feat_sliceNum))
    # Loop over each slice
    for i in range(feat_sliceNum):
        if mode == 'l2':
            # Calculate the L2 norm for each corresponding feature vector pair
            # and take the mean across all features within the slice
            dist[i, :] = np.mean(np.linalg.norm(set1[i] - set2, axis=3), axis=(1, 2))
        elif mode == 'cosine':
            dist = cosine_similarity_batch(set1, set2)


    """find the closest fixed feature for each moving feature"""
    if mode == 'l2' or mode == 'l2_align':
        closest_feature = np.argmin(dist, axis=1)
    else:
        closest_feature = np.argmax(dist, axis=1)
    return closest_feature

