# Plánování v umělé inteligenci

Workshop na *Poznej FI 2018*
o základních algoritmech pro plánování v umělé inteligenci:
hladové plánování, generuj a testuj,
DFS, BFS, UCS a A\*.
GitHub umí [připravené notebooky](./00-planning-in-AI.ipynb)
renderovat, ale interaktivní verze je lepší.
V repozitáři najdete i [vzorová řešení](./solutions.py).

## Start na FI

1. Otevřete terminál: `Ctrl+Alt+t`

2. Stáhněte si repozitář:

        $ git clone https://github.com/effa/ai-search-workshop.git

3. Přidejte modul s potřebnými balíčky pro Python 3:

        $ module add python3

4. Otevřete připravený *jupyter notebook*:

        $ jupyter notebook

## Start doma

1. Stáhněte si repozitář (tlačítko "Clone or download").

2. Nainstalujte si Python 3
   a balíčky uvedené v [requirements.txt](./requirements.txt),
   např. pomocí [Anacondy](https://www.anaconda.com/download).
   Balíčky v Anacondě nainstalujete buď klikáním v grafickém rozhraní *Anaconda Navigator*, nebo v přikazové řádce:

        $ conda install ipywidgets=7.1.0 jupyter=1.0.0 matplotlib=2.1.2 numpy=1.14.0

   (Alternativní možnosti instalace viz [Makefile](./Makefile).
   Pokud nepoužijte Anacondu, bude ještě potřeba aktivovat rozšíření
   pro interaktivní widgety. Je to jeden příkaz, jehož konkrétní podoba závisí
   na použitém způsobu instalace balíčků, viz Makefile.
   Kdybyste narazili na problém, tak mi napište a vyřešíme ho.)

3. Otevřete připravený *jupyter notebook*:

        $ jupyter notebook
