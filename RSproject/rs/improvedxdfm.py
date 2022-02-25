

from rs.models.improvedxdfm import ImprovedExtremeDeepFactorizationMachineModel

def get_improved_xdfm_model(layer_dimension = 16, dropout = 0.2, num_of_cim_layers=2,
              mlp_layer_size = 2):


    fields_dimension = [(73517 ,16), (34476 ,16), (7 ,16), (1819 ,16), (20 ,5), (1013918 ,8),
                        (44 ,4), (44 ,4), (44 ,4), (44 ,4), (14 ,4)]


    cross_layer_sizes = [layer_dimension for _ in range(num_of_cim_layers)]
    mlp_layer_sizes = [layer_dimension for _ in range(mlp_layer_size)]

    return ImprovedExtremeDeepFactorizationMachineModel(
        fields_dimension, cross_layer_sizes=cross_layer_sizes,
        split_half=False, mlp_dims=mlp_layer_sizes,
        dropout=dropout)