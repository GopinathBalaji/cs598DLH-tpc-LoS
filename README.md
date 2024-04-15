# cs598DLH-tpc-LoS

To load TPC-LoS, TPC-multitask, LSTM, and Transformer model, navigate to bdeleon2 and start a local runtime. You can connect to colab with the notebook and run the models. Must be on Python 3.6 and meet the `requirements.txt`. Make sure the paths.json is set correctly. Currently only trained on MIMIC.

change log:

changed `initialise_tpc_arguments` function of `models/initialise_arguments.py` to include `parser.add_argument('--model_type',default='tpc',type=str,help='can be either tpc, temp_only, or pointwise_only')`


changed line 444 of `models/tpc_model.py` in `temp_pointwise` function from:

`next_X = X_combined.view(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)  # B * ((F + Zt + point_size) * (1 + temp_kernels)) * T`

to:

`next_X = X_combined.reshape(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)  # B * ((F + Zt + point_size) * (1 + temp_kernels)) * T`
