# Modeling higher-order epistemic reasoning

This repository contains materials for the paper "Does Amy know Ben knows you know your cards? A computational model of higher-order epistemic reasoning". 

## Data
Columns:
1. "age": the subject's age.              
2. "AmyResponse": Amy's response (I know/I don't know).         
3. "answer": the subject's answer (what cards she holds. If response is "I don't know", encoded as '').   
4. "BenResponse": Ben's response (I know/I don't know).         
5. "cards": game state.               
6. "cards_iso": game states up to equivalence, e.g., 88AAAA is encoded as AA8888.          
7. "corAnswer": the subject's correct response (I know or I don't know).           
8. "exp_time": how long the subject took to finish the experiment, excluding the demographic form.               
9. "gender": the subject's gender.                         
10. "inference_level": the minimum level l required to guarantee a SUWEB model (with no stochasticity) with level >= l can solve the game.
11. "number": the number guess in the p-beauty contest.                  
12. "order": game order.               
13. "outcome": won or lost the current game.          
14. "outcomeArray": the correct annoucements by all players of a game (a list of lists), representing rounds in a game.        
15. "phase": game number.                  
16. "points": how many games the subject won out of ten games.              
17. "response": the subject's actual response (I know/I don't know).          
18. "round": the round number of a data point.                
19. "RT": the subject's reaction time for each round (counting from the onset of the previous announcement).  
20. "should_know": at which turn should the subject know her card (9 turns total, 10 means she never should know).
21. "subj": the subject ID.       
22. "numRound": the maximum number of round the game reaches if the subject keeps responding I don't know.

## Dependencies
Python: 3.7.6
- pandas 1.1.0
- numpy 1.19.1
- scipy 1.5.0
- matplotlib 3.2.2
- networkx 2.5

R: 4.0.2
- Metrics 0.1.4
- tidyverse 1.3.0
- plotrix 3.7.8

## Files to run
- fitting.py
- simulation.py
- results_visualization.Rmd

## src
- different computational models
- modeling procedures
- utils for Python and R