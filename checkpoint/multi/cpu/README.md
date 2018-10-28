# CPU Version

These checkpoints were trained on CPU (i.e pytorch device='cpu'). You should be able to map it to gpu using following code 
```
torch.load('my_file.pt', map_location={'cpu': 'cuda:0'})
```

or simply use cpu to review its performance.