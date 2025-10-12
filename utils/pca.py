import torch

def pca_lowrank_transform(all_features, n_components, mat=False):
    # Convert input features to a PyTorch tensor if not already
    all_features_tensor = torch.tensor(all_features, dtype=torch.float32)

    # Perform PCA using torch.pca_lowrank
    U, S, V = torch.pca_lowrank(all_features_tensor, q=n_components)

    # Compute the reduced representation by projecting the data onto the principal components
    # Note: The original data is projected onto the principal components to get the reduced data
    reduced_data = torch.matmul(all_features_tensor, V[:, :n_components])

    # Step 1: Square the singular values to get the eigenvalues
    eigenvalues = S.pow(2)
    # Step 2: Calculate the total variance
    total_variance = eigenvalues.sum()
    # Step 3: Normalize each eigenvalue to get the proportion of variance
    proportion_variance_explained = eigenvalues / total_variance

    if mat:
        return reduced_data, proportion_variance_explained, V
    return reduced_data, proportion_variance_explained