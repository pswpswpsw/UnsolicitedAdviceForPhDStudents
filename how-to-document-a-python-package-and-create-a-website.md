# Introduction

At some point in your Ph.D., you might want to open source your research and let others use your tools so as to maximize your impact to the world maximally! Only citation of your work does not matter.. People might cite you only for the sake of literature review. It is the people who are using your algorithm actually admitting your contributions! 

Imagine now you have already built a Python package just like my works: 
1. `pykoopman`: https://github.com/dynamicslab/pykoopman
2. `nif`: https://github.com/pswpswpsw/nif

Now you want to document your code. There are two ways to do so.

## 1. Local compiled a website


1. use ChatGPT to document the entire code 

2. use `sphinx` to create documented HTML. 

3. properly build the HTML website locally, then upload it via Github Page to github.

Example: [nif](https://github.com/pswpswpsw/nif). 

## 2. Use `readthedocs`

[Readthedocs](https://about.readthedocs.com/?ref=readthedocs.org) provide excellent service to host online documentation. It requires a bit tuning but fortunately I have paved the way for you already. Remember that you need to register an account in order to setup the online documentation.

Take a look at my `pykoopman` package. https://github.com/dynamicslab/pykoopman  It contains 
- a yaml to configure for readthedocs [readthedocs.yaml(https://github.com/dynamicslab/pykoopman/blob/master/.readthedocs.yaml)
- an index `.rst` file - the main page of your documentation https://github.com/dynamicslab/pykoopman/tree/master/docs
- an `conf.py`, - the configuration script for sphinx, which is used my readthedocs to render the documentation.

