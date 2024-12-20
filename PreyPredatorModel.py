from queue import PriorityQueue
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

#confidence intervals
def confidence_interval(data, confidence=0.90):
    mean_val = np.mean(data)
    sem = st.sem(data)  # Standard error of the mean
    interval = st.t.interval(confidence, len(data)-1, loc=mean_val, scale=sem)
    return interval

class State:
    def __init__(self, N_FemaleC, N_MaleC, N_FemaleH, N_MaleH):
        self.N_FemaleC=N_FemaleC
        self.N_MaleC=N_MaleC
        self.N_FemaleH=N_FemaleH
        self.N_MaleH=N_MaleH  
    def notAbsorbing(self):
        if (self.N_FemaleC==0 and self.N_MaleC==0 and self.N_FemaleH==0 and self.N_MaleH==0):
            return False
        else:
            return True

#DIFFERENT EVENTS:
# CARNIVOROUS REPRODUCTION
# HERBIVOROUS REPRODUCTION
# CARNIVOROUS KILLS HERBIVOROUS
# DEATH OF HERBIVOROUS
# DEATH OF CARNIVOROUS

def carnivorous_reproduction(state):
    if state.N_FemaleC>0 and state.N_MaleC>0:
            
        # Generate random number of new carnivores (between 1 and 3)
        possible_offspring = [1, 2, 3]  # Values of the new born
        probabilities = [0.35, 0.35, 0.2]  # Probabilità per ogni valore
        
        # Use random.choices to select according to weights
        num_new_carnivores = random.choices(possible_offspring, probabilities)[0]
        # Determine how many are female (50% chance for each)
        num_female = sum(random.random() < 0.5 for _ in range(num_new_carnivores))
        num_male = num_new_carnivores - num_female

        # Update the state
        state.N_FemaleC += num_female
        state.N_MaleC += num_male

        # Print debug information (optional)
        # print(f"[Time {time}] Carnivorous Reproduction: +{num_female} F, +{num_male} M")
        #print(f"Updated State: {state.N_FemaleC} FemaleC, {state.N_MaleC} MaleC")

def herbivorous_reproduction(state):
    if state.N_FemaleH>0 and state.N_MaleH>0:
        # Generate random number of new carnivores (between 1 and 3)
        possible_offspring = [1, 2, 3]  # Values of the new born
        probabilities = [0.2, 0.35, 0.35]  # Probability for each value
        
        # Use random.choices to select according to weights
        num_new_herbivorous = random.choices(possible_offspring, probabilities)[0]

        # Determine how many are female (50% chance for each)
        num_female = sum(random.random() < 0.5 for _ in range(num_new_herbivorous))
        num_male = num_new_herbivorous - num_female

        # Update the state
        state.N_FemaleH += num_female
        state.N_MaleH += num_male

        # Print debug information (optional)
        #print(f"[Time {time}] Herbivorous Reproduction: +{num_female} F, +{num_male} M")
        #print(f"Updated State: {state.N_FemaleH} FemaleH, {state.N_MaleH} MaleH")

def carnivorous_kills_herbivorous(state):
    # Check if there are carnivores and herbivores available
    if state.N_FemaleC + state.N_MaleC > 0 and state.N_FemaleH + state.N_MaleH > 0:
        # Randomly select if the prey can survive or not
        if random.random() < 0.5:
            # Randomly select whether if the killed individual is a female or a male, in case there is only one gender the death is that gender 
            if state.N_FemaleH>0 and state.N_MaleH>0:
                if random.random() < 0.5:
                    state.N_FemaleH -= 1
                else:
                    state.N_MaleH -= 1
            elif state.N_FemaleH>0 and state.N_MaleH==0:
                state.N_FemaleH -=1
            elif state.N_FemaleH==0 and state.N_MaleH>0:
                state.N_MaleH -=1
        

def death_carnivorous(state):
    if state.N_FemaleC + state.N_MaleC > 0:
        # Randomly select whether if the death is a female or a male, in case there is only one gender the death is that gender 
        if state.N_FemaleC>0 and state.N_MaleC>0:
            if random.random() < 0.5:
                state.N_FemaleC -= 1
            else:
                state.N_MaleC -= 1
        elif state.N_FemaleC>0 and state.N_MaleC==0:
            state.N_FemaleC -=1
        elif state.N_MaleC>0 and state.N_FemaleC==0:
            state.N_MaleC -=1

def death_herbivorous(state):
   if state.N_FemaleH + state.N_MaleH > 0:
        # Randomly select whether if the death is a female or a male, in case there is only one gender the death is that gender 
        if state.N_FemaleH>0 and state.N_MaleH>0:
            if random.random() < 0.5:
                state.N_FemaleH -= 1
            else:
                state.N_MaleH -= 1
        elif state.N_FemaleH>0 and state.N_MaleH==0:
            state.N_FemaleH -=1
        elif state.N_MaleH>0 and state.N_FemaleH==0:
            state.N_MaleH -=1

def updateRates(state):
    global rates, thresholds
    # Total population
    total_carnivores = state.N_FemaleC + state.N_MaleC
    total_herbivores = state.N_FemaleH + state.N_MaleH

    # Thresholds
    T_H = thresholds["herbivore_competition_threshold"]
    R_C = thresholds["carnivore_competition_threshold"]
    
    # ---------------------
    # Riproduzione carnivori
    # ---------------------
    if state.N_FemaleC > 0 and state.N_MaleC > 0 :
        rates["Carnivorous Reproduction"] = initial_rates["Carnivorous Reproduction"] * \
                                        state.N_FemaleC * state.N_MaleC
    else:
        rates["Carnivorous Reproduction"] = 0

    # ---------------------
    # Reproducing herbivores
    # ---------------------
    if state.N_FemaleH > 0 and state.N_MaleH > 0:
        rates["Herbivorous Reproduction"] = initial_rates["Herbivorous Reproduction"] * \
                                        state.N_FemaleH * state.N_MaleH
    else:
        rates["Herbivorous Reproduction"] = 0

    # ---------------------
    ### Killing herbivores by carnivores
    # ---------------------
    if total_herbivores > 0 and total_carnivores > 0:
        rates["Carnivorous kills Herbivorous"] = initial_rates["Carnivorous kills Herbivorous"] * \
                                                 total_carnivores * total_herbivores
    else:
        rates["Carnivorous kills Herbivorous"] = 0

    # ---------------------
    ### Herbivore death (intra-species competition)
    # ---------------------
    if total_herbivores > 0:
        if total_herbivores > T_H:  # Competition above the threshold
            rates["Death of Herbivorous"] = initial_rates["Death of Herbivorous"] * (total_herbivores)**2
        else:  # Mortalità naturale sotto soglia
            rates["Death of Herbivorous"] = initial_rates["Death of Carnivorous"] * total_herbivores
    else:
       rates["Death of Herbivorous"] = 0

    # ---------------------
    ### Carnivore death (intra-species competition + prey shortage)
    # ---------------------
    if total_carnivores > 0:
        ratio = total_carnivores / max(total_herbivores, 1)  # Ratio of carnivores to herbivores
        if ratio > R_C:  # Ratio above the threshold
            rates["Death of Carnivorous"] = initial_rates["Death of Carnivorous"] * total_carnivores**2
        else:
            rates["Death of Carnivorous"]= initial_rates["Death of Carnivorous"] * total_carnivores
    else:
        rates["Death of Carnivorous"] = 0




print()
print("Welcome to Prey/Predator simulation, there are two modes:\n"
      "\n"
      "Mode A:\t One single simulation, with this starting populations:\t N_FemaleC=50, N_MaleC=50, N_FemaleH=100, N_MaleH=100\n"
      "Mode B:\t 10 simulations, then confidence intervals of some metrics\n"
      "\n"
      "The initial rates, the threesholds and probabilities of making 1,2 or 3 child per reproduction are already set (but if you want feel free to change)\n"
      "They have been chosen to make the simulation as balanced as possible")

Mode=input("Insert A for a simulation and B for 10 simulations and confidence intervals (execution time can be > 5 min) ").upper()
if Mode=="A":

    #Simulation variables
    time=0
    FES=PriorityQueue()
    #Starting rates
    thresholds = {
        'herbivore_competition_threshold': 3000,  # Competition threshold for herbivores
        'carnivore_competition_threshold': 1,  # Competition threshold for carnivores (relative to herbivores)
    }
    initial_rates = {
        "Herbivorous Reproduction": 1,       ##lambda1   
        "Carnivorous Reproduction": 1,       ##lambda2
        "Death of Carnivorous": 1,           ##lambda3
        "Death of Herbivorous": 1,           ##lambda4
        "Carnivorous kills Herbivorous":0.4  ##lambda5
    }
    rates={
        "Herbivorous Reproduction": 1,
        "Carnivorous Reproduction": 1,
        "Death of Carnivorous": 1,
        "Death of Herbivorous": 1,
        "Carnivorous kills Herbivorous": 0.4
    }
    # Create initial state
    state = State(N_FemaleC=50, N_MaleC=50, N_FemaleH=100, N_MaleH=100)

    # Initialize events
    for event_type in rates.keys():
        rate = rates[event_type]
        if rate > 0:
            next_time = random.expovariate(rate)
            FES.put((next_time, event_type))
    sample_interval = 100  # For example, every 100 events
    event_count = 0        # Event counter
    samples = []

    while event_count < 5000000 and state.notAbsorbing():
        time, event_type = FES.get()  # Get the next event
        
        # Execute event
        if event_type == "Carnivorous Reproduction":
            carnivorous_reproduction(state)
        elif event_type == "Herbivorous Reproduction":
            herbivorous_reproduction(state)
        elif event_type == "Carnivorous kills Herbivorous":
            carnivorous_kills_herbivorous(state)
        elif event_type == "Death of Carnivorous":
            death_carnivorous(state)
        elif event_type == "Death of Herbivorous":
            death_herbivorous(state)
        #Update rates
        updateRates(state)
        # Update FES with new events
        for event_type in rates.keys():
            rate = rates[event_type]
            if rate > 0:
                next_time = time + random.expovariate(rate)
                FES.put((next_time, event_type))
                #print(f"Scheduled {event_type} at {next_time} Rate {rate}")
        # Increase event counter
        event_count += 1
        
        if event_count % sample_interval == 0:
            samples.append((event_count,state.N_FemaleC + state.N_MaleC, state.N_FemaleH + state.N_MaleH))
            print(f"[Sample {event_count // sample_interval}], FC={state.N_FemaleC}, MC={state.N_MaleC}, FH={state.N_FemaleH}, MH={state.N_MaleH}")
    # Extract data from the samples for the graph
    event_counts, carnivores, herbivores = zip(*[(x[0], x[1], x[2]) for x in samples])

    # Graph creation
    plt.figure(figsize=(10, 6))
    plt.plot(event_counts, carnivores, label="Carnivores", color="red")
    plt.plot(event_counts, herbivores, label="Herbivores", color="green")
    plt.xlabel("Event Count")
    plt.ylabel("Population")
    plt.title("Population Dynamics Over Event Count")
    plt.legend()
    plt.grid()
    plt.show()
elif Mode=="B":
    NUM_RUNS=10
    CONF_LEVEL=0.9
    run=1
    HerbReproductionsEND=[]
    CarnReproductionsEND=[]
    HerbCarnEncountersEND=[]
    HerbDeathsEND=[]
    CarnDeathsEND=[]
    for i in range(NUM_RUNS):
        print(f"Run number {run}")

        #Simulation variables
        time=0
        FES=PriorityQueue()
        #Starting rates
        thresholds = {
            'herbivore_competition_threshold': 3000,  # Competition threshold for herbivores
            'carnivore_competition_threshold': 1,  # Competition threshold for carnivores (relative to herbivores)
        }

        initial_rates = {
            "Herbivorous Reproduction": 1,      ##lambda1
            "Carnivorous Reproduction": 1,      ##lambda2
            "Death of Carnivorous": 1,          ##lambda3
            "Death of Herbivorous": 1,          ##lambda4
            "Carnivorous kills Herbivorous":0.4 ##lambda5
        }
        rates={
            "Herbivorous Reproduction": 1,
            "Carnivorous Reproduction": 1,
            "Death of Carnivorous": 1,
            "Death of Herbivorous": 1,
            "Carnivorous kills Herbivorous": 0.4
        }
        HerbReproductions=0
        CarnReproductions=0
        HerbCarnEncounters=0
        HerbDeaths=0
        CarnDeaths=0
        # Create initial state
        state = State(N_FemaleC=50, N_MaleC=50, N_FemaleH=100, N_MaleH=100)

        # Initialize events
        for event_type in rates.keys():
            rate = rates[event_type]
            if rate > 0:
                next_time = random.expovariate(rate)
                FES.put((next_time, event_type))
        sample_interval = 100  # For example, every 100 events
        event_count = 0        # Event counter
        samples = []

        while event_count < 5000000 and state.notAbsorbing():
            time, event_type = FES.get()  # Get the next event
            
            # Esegui l'evento
            if event_type == "Carnivorous Reproduction":
                carnivorous_reproduction(state)
                CarnReproductions+=1
            elif event_type == "Herbivorous Reproduction":
                herbivorous_reproduction(state)
                HerbReproductions+=1
            elif event_type == "Carnivorous kills Herbivorous":
                HerbCarnEncounters+=1
                carnivorous_kills_herbivorous(state)
            elif event_type == "Death of Carnivorous":
                CarnDeaths+=1
                death_carnivorous(state)
            elif event_type == "Death of Herbivorous":
                HerbDeaths+=1
                death_herbivorous(state)
            #updating rates
            updateRates(state)
            # Update FES with new events
            for event_type in rates.keys():
                rate = rates[event_type]
                if rate > 0:
                    next_time = time + random.expovariate(rate)
                    FES.put((next_time, event_type))        
            # Increase event counter
            event_count += 1
        print(f"END STATE: FC={state.N_FemaleC}, MC={state.N_MaleC}, FH={state.N_FemaleH}, MH={state.N_MaleH}")
        HerbReproductionsEND.append(HerbReproductions)
        CarnReproductionsEND.append(CarnReproductions)
        HerbCarnEncountersEND.append(HerbCarnEncounters)
        HerbDeathsEND.append(HerbDeaths)
        CarnDeathsEND.append(CarnDeaths)
        run+=1
    ci_HerbReproductions=confidence_interval(HerbReproductionsEND)
    ci_CarnReproductions=confidence_interval(CarnReproductionsEND)
    ci_HerbCarnEncounters=confidence_interval(HerbCarnEncountersEND)
    ci_HerbDeaths=confidence_interval(HerbDeathsEND)
    ci_CarnDeaths=confidence_interval(CarnDeathsEND)
    print("Confidence intervals:")
    print(f"Number of Herbivorous Reproductions encounters at {CONF_LEVEL*100}% CI: {ci_HerbReproductions}")
    print(f"Number of Carnivorous Reproductions encounters at {CONF_LEVEL*100}% CI: {ci_CarnReproductions}")
    print(f"Number of Herbivorous and Carnivorous encounters at {CONF_LEVEL*100}% CI: {ci_HerbCarnEncounters}")
    print(f"Number of Herbivorous Deaths at {CONF_LEVEL*100}% CI: {ci_HerbDeaths}")
    print(f"Number of Carnivorous Deaths at {CONF_LEVEL*100}% CI: {ci_CarnDeaths}")
    
    # Dati
    categories = [
        "Herbivorous Reproductions", 
        "Carnivorous Reproductions", 
        "Herb & Carn Encounters", 
        "Herbivorous Deaths", 
        "Carnivorous Deaths"
    ]
    conf_intervals = [
        ci_HerbReproductions,
        ci_CarnReproductions,
        ci_HerbCarnEncounters,
        ci_HerbDeaths,
        ci_CarnDeaths
    ]

    # Calculation of minimum and maximum values
    mins = [low for low, high in conf_intervals]
    maxs = [high for low, high in conf_intervals]

    x = np.arange(len(categories))  # Category positions
    plt.figure(figsize=(12, 8))
    plt.barh(x, [high - low for low, high in conf_intervals], left=mins, color='lightblue', edgecolor='black', alpha=0.7)
    plt.yticks(x, categories,fontsize=7)
    plt.xlabel("Number of Events")
    plt.title(f"Confidence Intervals at {90}% CI")
    plt.xlim(0, max(maxs) * 1.1)  # Axis x starts at 0 and has a margin

    # Adding value labels
    for i, (low, high) in enumerate(conf_intervals):
        plt.text(high, i, f"{int(high)}", va='center', ha='left', fontsize=9)
        plt.text(low, i, f"{int(low)}", va='center', ha='right', fontsize=9)

    plt.show()

else:
    print("Not valid input!")
    exit(1)