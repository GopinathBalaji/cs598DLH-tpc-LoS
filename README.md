# cs598DLH-tpc-LoS
change log:
changed `initialise_tpc_arguments` function of `initialise_arguments.py` to include `parser.add_argument('--model_type',default='tpc',type=str,help='can be either tpc, temp_only, or pointwise_only')`


changed line 444 of `tpc_model.py` in `temp_pointwise` function from:
`next_X = X_combined.view(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)  # B * ((F + Zt + point_size) * (1 + temp_kernels)) * T`
to:
`next_X = X_combined.reshape(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)  # B * ((F + Zt + point_size) * (1 + temp_kernels)) * T`
