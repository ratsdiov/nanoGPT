# From https://medium.com/@kl.yap/replicating-tinystories-paper-38839d03ec81

# TODO - Make checkpoint be timestamp and parameter named

out_dir = "out-tinystories-train"
eval_interval = 1000  # keep frequent because we'll overfit
eval_iters = 250
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "tinystories-train"
wandb_run_name = "mini-gpt"

dataset = "tinystories"
# dataset = "tinystories-1k"
gradient_accumulation_steps = 3
batch_size = 32
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 4
n_embd = 384
dropout = 0.2

# Most of my changes are here
learning_rate = 5e-4  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 5e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
