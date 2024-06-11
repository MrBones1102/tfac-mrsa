import importlib
from matplotlib.pyplot import close

figureSuffixes = ["2", "4", "5", "6", "A1", "A2", "S1", "S2", "S3",
                  "S4", "S5"]

figModules = []
for _ in figureSuffixes:
    figModules.append(importlib.import_module("figure"+_))
    # print(figModules)


for figure in figModules[2:4]:
    if hasattr(figure, "makeFigure"):
        print(f"Calling: {figure.__name__}.makefigure()")
        print(figure)
        fig = figure.makeFigure()
        fig.savefig(f"./JamesIter{figure.__name__}.png")


"""
There's a bug with figure6.makeFigure(). It fails to make the figure if
the loop had just previously generated figure5. Somehow, there's a leftover
'<' somewhere in the data stream which causes predict.py to fail

Figure S5 takes a long time to make with no print statements indicating
progress. 
"""
