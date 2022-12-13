# Cal_Poly_Senior_Project

## Before starting...

### Dependencies

Use python version 3 (latest version recommended)

Use `pip install <module>` on the following modules (documentation linked):
- [pygad](https://pypi.org/project/pygad/)
- [stockfish](https://pypi.org/project/stockfish/)
- [numpy](https://numpy.org/install/)
- [chess](https://python-chess.readthedocs.io/en/latest/)

For stockfish especially, read the documentation on how to set it up. 
Your stockfish location needs to be edited within the `final_project.py` file to run it properly

---

### Files By Type

  > ### PDF
  > - `Senior_Project_Proposal.pdf`: an initial proposal submitted for the project.
  > - `SeniorProjectQ1Report.pdf`: a report about the work done on the project during the first quarter (spring 2022).
  > - `Initial_Middlegame_Issues.pdf`: a file describing the issues I ran into with the [sunfish](https://github.com/thomasahle/sunfish) genetic algorithm implementation attempt.  

   > ### py
   > - `final_project.py`: the working final project implementation of a chess AI.
   > - `original_pst.py` and `pst.py`: files that contain the genetic algorithm implementation for finding optimal piece square talbes. These were built upon the [Andoma](https://github.com/healeycodes/andoma) chess engine.
   > - `analysis.py`: a program that tests n number of times whether or not a position or m pieces reduces to a solved endgame. The csv files are the results.  
   
   > ### csv
   > These files are all the results of the `analysis.py` program above
  
   > ### txt
   > These files are the best solution obtained from the `pst.py` program above and the solution fit value.
   
---

### Running the program

   Run 

   ```bash
   py <filename.py>
   ```

   For any of the files: `final_project.py`, `pst.py`, `analysis.py` (This one will provide usage instructions)

