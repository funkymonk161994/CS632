Short installation guide for Mac/Linux, using virtualenv.

```
$ cd ~
$ virtualenv --system-site-packages ~/my_env
$ source ~/my_env/bin/activate
$ git clone https://github.com/random-forests/cs632.git
$ cd cs632/setup
$ pip install -r requirements.txt
$ curl -O http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
$ tar xvf cifar-10-python.tar.gz
$ python extract_cats_dogs.py
```