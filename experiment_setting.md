# 加速效果
```
python main.py --graph com-dblp --method appr locgd aespappr aesplocgd --nodenum 50 --alpha 0.1 0.01 0.001 --eps 1e-8
python main.py --graph com-dblp --method appr locgd aespappr aesplocgd --nodenum 50 --alpha 0.1 0.06309573444801933 0.039810717055349734 0.025118864315095808 0.01584893192461114 0.01 0.006309573444801936 0.003981071705534978 0.002511886431509582 0.001584893192461114 --eps 1e-8
```
# R
```
python main.py --graph com-dblp --method aesparr --node 10 --alpha 0.01 --eps 1e-06
python main.py --method aespappr --node 20 --alpha 0.1 0.01 0.001 --eps 1e-6
python main.py --graph ogbn-papers100M --method aespappr --node 681 --alpha 0.1 0.01 0.001 --eps 1e-6
python main.py --graph ogb-mag240m --method aespappr --node 86714087 --alpha 0.01 --eps 1e-6
python main.py --graph ogb-mag240m --method aespappr --node 86714087 --alpha 0.001 --eps 1e-6
```
# grad_one vol/gamma
```
python main.py --method aespappr --nodenum 50 --alpha 0.1 --eps 1e-6
```

# convergence 
```
python main.py --graph com-dblp --method appr locgd appropt aespappr aesplocgd --node 20 --alpha 0.01 --eps 1e-7

python main.py --graph com-youtube --method appr locgd appropt aespappr aesplocgd --node 10 --alpha 0.01 --eps 1e-8

python main.py --graph com-friendster --method appr locgd appropt aespappr aesplocgd --node 46540436 --alpha 0.01 --eps 1e-10


python main.py --graph as-skitter --method appr locgd appropt aespappr aesplocgd --node 20 --alpha 0.01 --eps 1e-7
```

# init (finish)
```
python main.py --graph com-youtube --method xinit yinit zeroinit --node 20 --alpha 0.01 --eps 1e-10
```

# aspr locCH
```
python main.py --graph com-dblp --method aespappr aesplocgd --node 20 --alpha 0.01 --eps 1e-7
python main.py --graph com-dblp --method cheby fista aspr --node 20 --alpha 0.01 --eps 1e-7
```


# 加速方法的比较