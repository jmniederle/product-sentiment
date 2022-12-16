# Tweet sentiment
Take a look at `demo.ipynb` for an example of scraping, sentiment classification and visualisation of the results.

To train model run `sentiment_model/runner.py`. For wandb logging make sure to enter wandb username in 
`sentiment_model/wandb_user_name.txt` or disabling wandb logging by passing `wandb_logging=False` to `run_training`.

To calibrate model and save calibrated model run the notebook `sentiment_model/compute_decision_boundary.ipynb`