from queue import PriorityQueue
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import networkx as nx

flag=True
# This flag is used to enable the printing of a customer's journey:
# IF THE CUSTOMER ONLY NEEDS ONE LINE TO REACH THE DESTINATION
# Path
# Arrival time at station
# Boarding time on the train and station
# Disembarking time from the train and station

# IF THE CUSTOMER NEEDS TO MAKE A TRANSFER
# Path
# Arrival time at station
# Boarding time on the train and station
# Disembarking time for transfer and station
# Boarding time on the second train and station
# Disembarking time from the train and station


#confidence intervals
def confidence_interval(data, confidence=0.90):
    mean_val = np.mean(data)
    sem = st.sem(data)  # Standard error of the mean
    interval = st.t.interval(confidence, len(data)-1, loc=mean_val, scale=sem)
    return interval

def calculate_change_station(path):
    """
    Calcola la stazione di cambio per un cliente dato il percorso (path).
    Restituisce la prima stazione in cui avviene un cambio di treno.
    """
    current_train = None

    # Itera through the stations in the route
    for i in range(len(path) - 1):
        current_station = path[i]
        next_station = path[i + 1]

        # Find the train serving the current segment
        for train in trains:
            if current_station in train.path and next_station in train.path:
                current_train = train.id_train
                break

        # Check if the train changes to the next segment
        if i + 1 < len(path) - 1:
            next_next_station = path[i + 2]
            next_train = None
            for train in trains:
                if next_station in train.path and next_next_station in train.path:
                    next_train = train.id_train
                    break

            # If the train changes, return the change station
            if current_train != next_train:
                return next_station

    # No change necessary
    return None
class SimulationMeasure:
    def __init__(self):
        #stations measures
        self.stationA_customers = []
        self.stationB_customers = []
        self.stationC_customers = []
        self.stationD_customers = []
        self.stationE_customers = []
        self.stationF_customers = []
        self.stationG_customers = []
        self.stationH_customers = []
        self.stationI_customers = []
        #path measures
        self.path_times={}
        #trainsmeasure
        self.LineAC_customers=[]
        self.LineDF_customers=[]
        self.LineGI_customers=[]
        self.LineAG_customers=[]
        self.LineBH_customers=[]
        self.LineCI_customers=[]

    def print_average_times(self):
        for path_key in sorted(self.path_times.keys(), key=lambda x: tuple(sorted(x))):
            average_time = sum(self.path_times[path_key]) / len(self.path_times[path_key])
            print(f"Path {path_key}: Average time = {average_time:.2f}")

class ClientMeasure:
    def __init__(self, client_id, path,arrival_time, change_station):
        self.client_id = client_id
        self.path=path
        self.arrival_time = arrival_time
        self.change_station = change_station
        self.ascend = (None,None)
        self.change= (None,None)
        self.ascendChange=(None,None)
        self.descend = (None,None)
        self.total_delay=None

    def set_ascend(self, time,station):
        self.ascend= (time,station)
    def set_change(self,time,station):
        self.change=(time,station)
    def set_ascendChange(self,time,station):
        self.ascendChange=(time,station)
    def set_descend(self, time,station):
        self.descend=(time,station)

    def print_client_measure(self):
        output = f"Client{self.client_id} - Path {self.path} - Arrival: {self.arrival_time:.2f}, "
        output += f"Ascend: {self.ascend[0]:.2f} {self.ascend[1]}"

        #Add 'change' if it is not None
        if self.change[0] is not None:
            output += f", Change {self.change[0]:.2f} {self.change[1]}"

        # Add 'ascendChange' if it is not None
        if self.ascendChange[0] is not None:
            output += f", Second Ascend {self.ascendChange[0]:.2f} {self.ascendChange[1]}"

        # Add 'descend' (always present)
        output += f" Descend: {self.descend[0]:.2f} {self.descend[1]}"
        self.total_time=self.descend[0]-self.arrival_time

        path_key = tuple(self.path)  # Use a tuple as a key to represent the path
        if path_key not in data.path_times:
            data.path_times[path_key] = []  # Create a new list if it does not exist
            data.path_times[path_key].append(self.total_time)


        # Print result
        print(output)

class Client:
    def __init__(self, id, arrival_time,path,change_station,last_station):
        self.id = id
        self.arrival_time = arrival_time
        self.path = path if path is not None else []
        self.last_station=last_station
        self.measure=None
        self.change_station = change_station  # Where the client needs to change trains, if needed
        self.already_changed=False
        self.next_station = self.path[1] if len(self.path) > 1 else None
    def get_change_station(self):
        return self.change_station
    def set_already_changed(self):
        self.already_changed=True
    def checkTrain(self, train_path):
        return self.next_station in train_path if self.next_station else False
    def next(self):
        """Aggiorna il cliente alla prossima fermata nel percorso."""
        if len(self.path) > 1:  # Se ci sono ancora fermate
            self.path.pop(0)  # Rimuovi la fermata attuale
            self.next_station = self.path[1] if len(self.path) > 1 else None
        else:
            self.next_station = None  # Nessuna fermata successiva
class Station:
    def __init__(self, station_id):
        self.station_id = station_id
        self.customers = []  #Clients into the station

    def add_customer(self, customer):
        self.customers.append(customer)

    def remove_customer(self,id):
        for client in self.customers:
            if client.id == id:
                self.customers.remove(client)
                return id  #If you want to return the removed client
        return None 

class Train:
    def __init__(self, id_train, capacity, path, time,station,last_station):
        self.id_train = id_train
        self.capacity = capacity
        self.path=path
        self.now=(time,station)
        self.onboard_customers = []
        self.last_station = last_station
        self.delay = 0  # Ritardo calcolato in train_arrival
    def __lt__(self, other):
        return self.id_train < other.id_train  # Confronto basato sull'ID del treno

    def add_client(self, client):
        if len(self.onboard_customers) < self.capacity:
            self.onboard_customers.append(client)
            return True
        return False
    def remove_client(self,client):
        self.onboard_customers.remove(client)

    def advance(self):
        '''Advance the train to the next step in its journey.'''
        # Trova l'indice della stazione corrente
        current_time=self.now[0]
        current_index = self.path.index(self.now[1])
        
        if not self.last_station:
            # Se il treno sta andando avanti
            if current_index + 1 < len(self.path):
                next_station = self.path[current_index + 1]
            else:
                # Ha raggiunto l'ultima stazione, inizia a tornare indietro
                self.last_station = True
                next_station = self.path[current_index - 1]  # Torna indietro
        else:
            # Se il treno sta tornando indietro
            if current_index - 1 >= 0:
                next_station = self.path[current_index - 1]
            else:
                # Ha raggiunto la prima stazione, torna avanti
                self.last_station = False
                next_station = self.path[current_index + 1]  # Torna avanti

        # Aggiorna lo stato del treno (tempo corrente, nuova stazione)
        self.now = (current_time + self.delay, next_station)
        ##print(f"Train {self.id_train} moved to station {next_station} at time {self.now[0]:.2f}")
            

#MAP OF THE METRO
map = [
    ['A', 'B', 'C'],
    ['D', 'E', 'F'],
    ['G', 'H', 'I']
]
# Graph creation
G = nx.Graph()
# We define connections (only horizontal and vertical adjacencies)
edges = [
    ('A', 'B'), ('B', 'C'),
    ('D', 'E'), ('E', 'F'),
    ('G', 'H'), ('H', 'I'),
    ('A', 'D'), ('D', 'G'),
    ('B', 'E'), ('E', 'H'),
    ('C', 'F'), ('F', 'I')
]

G.add_edges_from(edges)
def arrivalClient(time, FES):
    global idClient,stations
    departure_stations=['A','B','C',
                        'D','E','F',
                        'G','H','I']
    
    departure_station=random.choices(departure_stations,
                            weights=[0.1,0.1,0.1,
                                     0.1,0.2,0.1,
                                     0.1,0.1,0.1],k=1)[0]
    departure_stations.remove(departure_station)
    stations_destinations = departure_stations
    station_destination=random.choices(stations_destinations,
                                      weights=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125],k=1)[0]
    path = nx.shortest_path(G, source=departure_station, target=station_destination)
    change_station = calculate_change_station(path)
    customer=Client(idClient,time,path.copy(),change_station,station_destination)
    measure=ClientMeasure(idClient, path.copy(),time, change_station)
    customer.measure=measure
    stations[departure_station].add_customer(customer)
    ##print(f"Client{idClient} arrive in station {departure_station}, path: {path}")
    inter_arrival = random.expovariate(1.0 / ARRIVAL)
    FES.put((time + inter_arrival, "arrival_client"))
    idClient+=1

def train_arrival(time, train,FES):
    current_station = train.now[1]
    ##print(f"Train {train.id_train} arrived at station {current_station} at time {time:.2f}")
    ##print(f"On board customers: {len(train.onboard_customers)}, Station{current_station} waiting customers: {len(stations[current_station].customers)}")

    #Customers to go down
    disembarking_clients = []
    for client in train.onboard_customers:
        client.next()
        ##print(f"Client{client.id} - Current station: {current_station},  Path: {client.measure.path}")

        #Check if it should go down (destination or change)
        if client.get_change_station():
            if client.already_changed:
                if client.last_station==current_station:
                    disembarking_clients.append(client)
            else:
                if current_station == client.get_change_station() or client.last_station==current_station:
                    disembarking_clients.append(client)
                    client.set_already_changed()
        else:
            if client.last_station==current_station:
                    disembarking_clients.append(client)
    
    num_disembarking_clients=len(disembarking_clients)
    for client in disembarking_clients:
        train.remove_client(client)  # Remove from train
        if client.last_station==current_station:  
            client.measure.set_descend(time,current_station)
            #Enable the print of customer path only for the first run (less heavy stdout)
            if flag:
                client.measure.print_client_measure()
        if client.change_station==current_station:
            stations[current_station].add_customer(client)
            client.measure.set_change(time,current_station)
    
    #Customers to go up
    boarding_clients = []
    for client in stations[current_station].customers:
        if client.checkTrain(train.path):
            boarding_clients.append(client)

    num_boarding = 0
    for client in boarding_clients:
        if train.add_client(client):
            stations[current_station].remove_customer(client.id)
            if current_station==client.change_station:
                client.measure.set_ascendChange(time,current_station)
            else:
                client.measure.set_ascend(time,current_station)
            num_boarding += 1
            ##print(f"Client{client.id} boarded train {train.id_train} at station {current_station}.")

    disembark_delay = random.expovariate(1 / max(1, num_disembarking_clients * 0.5))  # Delay proportional to customers descending
    boarding_delay = random.expovariate(1 / max(1, num_boarding * 0.3))  # Delay proportional to customers going up
    random_delay = random.expovariate(1 / 2.0)  # Random delay

    train.delay =  disembark_delay + boarding_delay + random_delay 
    ##print(f"Train {train.id_train} delay calculated: {train.delay:.2f}")

    #Get statistics on the number of users on board
    if (train.id_train==1 or train.id_train==7):
        data.LineAC_customers.append((len(train.onboard_customers)))
    if (train.id_train==2 or train.id_train==8):
        data.LineDF_customers.append((len(train.onboard_customers)))
    if (train.id_train==3 or train.id_train==9):
        data.LineGI_customers.append((len(train.onboard_customers)))
    if (train.id_train==4 or train.id_train==10):
        data.LineAG_customers.append((len(train.onboard_customers)))
    if (train.id_train==5 or train.id_train==11):
        data.LineBH_customers.append((len(train.onboard_customers)))
    if (train.id_train==6 or train.id_train==12):
        data.LineCI_customers.append((len(train.onboard_customers)))

    #Plan the next train arrival
    train.advance()
    FES.put((train.now[0], "train_arrival", train))


# Input parameters
ARRIVAL = 0.15
idClient=0
# Parameters for the runs and confidence intervals
NUM_RUNS = 30
CONF_LEVEL = 0.9
#Lists for average customer of station at each run
stationA_runs = []
stationB_runs = []
stationC_runs = []
stationD_runs = []
stationE_runs = []
stationF_runs = []
stationG_runs = []
stationH_runs = []
stationI_runs = []
#List for average customer of train line at each run
LineAC_runs=[]
LineDF_runs=[]
LineGI_runs=[]
LineAG_runs=[]
LineBH_runs=[]
LineCI_runs=[]

for i in range(NUM_RUNS):
    if i!=0:
        flag=False
    # Initialize stations
    stationA=Station('A')
    stationB=Station('B')
    stationC=Station('C')
    stationD=Station('D')
    stationE=Station('E')
    stationF=Station('F')
    stationG=Station('G')
    stationH=Station('H')
    stationI=Station('I')
    stations = {
        'A': stationA,
        'B': stationB,
        'C': stationC,
        'D': stationD,
        'E': stationE,
        'F': stationF,
        'G': stationG,
        'H': stationH,
        'I': stationI
    }

    LineAC=['A', 'B', 'C']
    LineDF=['D', 'E', 'F']
    LineGI=['G', 'H', 'I']
    LineAG=['A', 'D', 'G']
    LineBH=['B', 'E', 'H']
    LineCI=['C', 'F', 'I']

    #Initailize trains
    train1 = Train(1, 40, LineAC, 0, 'A',False)
    train2 = Train(2, 40, LineDF, 0, 'D',False)
    train3 = Train(3, 40, LineGI, 0, 'G',False)
    train4 = Train(4, 40, LineAG, 0, 'A',False)
    train5 = Train(5, 40, LineBH, 0, 'B',False)
    train6 = Train(6, 40, LineCI, 0, 'C',False)
    
    train7 = Train(7, 40, LineAC, 0, 'C',True)
    train8 = Train(8, 40, LineDF, 0, 'F',True)
    train9 = Train(9, 40, LineGI, 0, 'I',True)
    train10 = Train(10,40,LineAG, 0, 'G',True)
    train11= Train(11, 40,LineBH, 0, 'H',True)
    train12= Train(12, 40,LineCI, 0, 'I',True)
    
    trains=[train1,train2,train3,train4,train5,train6,train7,train8,train9,train10,train11,train12]

    #Simulation
    time=0
    data=SimulationMeasure()
    FES=PriorityQueue()
    FES.put((0, "arrival_client"))
    for train in trains:
        FES.put((0, "train_arrival", train))
    max_time=1000
    while time < max_time:
        time, event_type, *args = FES.get()
        if event_type == "arrival_client":
            arrivalClient(time, FES)
        elif event_type == "train_arrival":
            train = args[0]  # Retrieve the associated train
            train_arrival(time, train,FES)
            data.stationA_customers.append(len(stationA.customers))
            data.stationB_customers.append(len(stationB.customers)) 
            data.stationC_customers.append(len(stationC.customers)) 
            data.stationD_customers.append(len(stationD.customers))
            data.stationE_customers.append(len(stationE.customers))
            data.stationF_customers.append(len(stationF.customers))
            data.stationG_customers.append(len(stationG.customers))
            data.stationH_customers.append(len(stationH.customers))
            data.stationI_customers.append(len(stationI.customers))
            

    #Stations average customers
    average_customers_A = np.mean(data.stationA_customers)
    average_customers_B = np.mean(data.stationB_customers)
    average_customers_C = np.mean(data.stationC_customers)
    average_customers_D = np.mean(data.stationD_customers)
    average_customers_E = np.mean(data.stationE_customers)
    average_customers_F = np.mean(data.stationF_customers)
    average_customers_G = np.mean(data.stationG_customers)
    average_customers_H = np.mean(data.stationH_customers)
    average_customers_I = np.mean(data.stationI_customers)

    #Saving stats for confidence intervals
    stationA_runs.append(average_customers_A)
    stationB_runs.append(average_customers_B)
    stationC_runs.append(average_customers_C)
    stationD_runs.append(average_customers_D)
    stationE_runs.append(average_customers_E)
    stationF_runs.append(average_customers_F)
    stationG_runs.append(average_customers_G)
    stationH_runs.append(average_customers_H)
    stationI_runs.append(average_customers_I)
    '''
    print("Stations statistics:")
    print(f"Mean Customers Station A: {average_customers_A}")
    print(f"Mean Customers Station B: {average_customers_B}")
    print(f"Mean Customers Station C: {average_customers_C}")
    print(f"Mean Customers Station D: {average_customers_D}")
    print(f"Mean Customers Station E: {average_customers_E}")
    print(f"Mean Customers Station F: {average_customers_F}")
    print(f"Mean Customers Station G: {average_customers_G}")
    print(f"Mean Customers Station H: {average_customers_H}")
    print(f"Mean Customers Station I: {average_customers_I}")
    print()
    '''
    
    #Lines average customers
    average_lineAC_customers = np.mean(data.LineAC_customers)
    average_lineDF_customers = np.mean(data.LineDF_customers)
    average_lineGI_customers = np.mean(data.LineGI_customers)
    average_lineAG_customers = np.mean(data.LineAG_customers)
    average_lineBH_customers = np.mean(data.LineBH_customers)
    average_lineCI_customers = np.mean(data.LineCI_customers)

    #Saving stats for confidence intervals
    LineAC_runs.append(average_lineAC_customers)
    LineDF_runs.append(average_lineDF_customers)
    LineGI_runs.append(average_lineGI_customers)
    LineAG_runs.append(average_lineAG_customers)
    LineBH_runs.append(average_lineBH_customers)
    LineCI_runs.append(average_lineCI_customers)
    '''
    print("Lines Statistics:")
    print(f"Mean Customers LineAC: {average_lineAC_customers}")
    print(f"Mean Customers LineDF: {average_lineDF_customers}")
    print(f"Mean Customers LineGI: {average_lineCI_customers}")
    print(f"Mean Customers LineAG: {average_lineAG_customers}")
    print(f"Mean Customers LineBH: {average_lineBH_customers}")
    print(f"Mean Customers LineCI: {average_lineCI_customers}")
    print()
    '''

    #Enable statistics about every path average time only for the first run
    if (i==0):
        print("Path statistics:")
        data.print_average_times()

ci_A=confidence_interval(stationA_runs)
ci_B=confidence_interval(stationB_runs)
ci_C=confidence_interval(stationC_runs)
ci_D=confidence_interval(stationD_runs)
ci_E=confidence_interval(stationE_runs)
ci_F=confidence_interval(stationF_runs)
ci_G=confidence_interval(stationG_runs)
ci_H=confidence_interval(stationH_runs)
ci_I=confidence_interval(stationI_runs)

ci_LineAC=confidence_interval(LineAC_runs)
ci_LineDF=confidence_interval(LineDF_runs)
ci_LineGI=confidence_interval(LineGI_runs)
ci_LineAG=confidence_interval(LineAG_runs)
ci_LineBH=confidence_interval(LineBH_runs)
ci_LineCI=confidence_interval(LineCI_runs)

print("Confidence intervals for station metrics:")
print(f"Station A (Users in Station) at {CONF_LEVEL*100}% CI: {ci_A}")
print(f"Station B (Users in Station) at {CONF_LEVEL*100}% CI: {ci_B}")
print(f"Station C (Users in Station) at {CONF_LEVEL*100}% CI: {ci_C}")
print(f"Station D (Users in Station) at {CONF_LEVEL*100}% CI: {ci_D}")
print(f"Station E (Users in Station) at {CONF_LEVEL*100}% CI: {ci_E}")
print(f"Station F (Users in Station) at {CONF_LEVEL*100}% CI: {ci_F}")
print(f"Station G (Users in Station) at {CONF_LEVEL*100}% CI: {ci_G}")
print(f"Station H (Users in Station) at {CONF_LEVEL*100}% CI: {ci_H}")
print(f"Station I (Users in Station) at {CONF_LEVEL*100}% CI: {ci_I}")

print("\nConfidence intervals for line metrics:")
print(f"Line AC (Users on Line AC) at {CONF_LEVEL*100}% CI: {ci_LineAC}")
print(f"Line DF (Users on Line DF) at {CONF_LEVEL*100}% CI: {ci_LineDF}")
print(f"Line GI (Users on Line GI) at {CONF_LEVEL*100}% CI: {ci_LineGI}")
print(f"Line AG (Users on Line AG) at {CONF_LEVEL*100}% CI: {ci_LineAG}")
print(f"Line BH (Users on Line BH) at {CONF_LEVEL*100}% CI: {ci_LineBH}")
print(f"Line CI (Users on Line CI) at {CONF_LEVEL*100}% CI: {ci_LineCI}")

#Plotting some stats
runs = list(range(1, NUM_RUNS + 1))

# Creating the scatter plot for all stations.
plt.figure(figsize=(12, 8))

# Add each station as a separate series.
plt.scatter(runs, stationA_runs, label='Station A', color='blue', marker='o')
plt.scatter(runs, stationB_runs, label='Station B', color='green', marker='x')
plt.scatter(runs, stationC_runs, label='Station C', color='red', marker='s')
plt.scatter(runs, stationD_runs, label='Station D', color='purple', marker='^')
plt.scatter(runs, stationE_runs, label='Station E', color='orange', marker='D')
plt.scatter(runs, stationF_runs, label='Station F', color='pink', marker='*')
plt.scatter(runs, stationG_runs, label='Station G', color='cyan', marker='P')
plt.scatter(runs, stationH_runs, label='Station H', color='brown', marker='H')
plt.scatter(runs, stationI_runs, label='Station I', color='yellow', marker='v')

# Labels and title
plt.xlabel('Run Number')
plt.ylabel('Number of Users in Station')
plt.title('Scatter Plot of Users in Stations across Runs')
plt.legend()

# Layout and display
plt.tight_layout()
plt.show()


# Creating the scatter plot for all lines.
plt.figure(figsize=(12, 8))

# Add each line as a separate series
plt.scatter(runs, LineAC_runs, label='Line AC', color='blue', marker='o')
plt.scatter(runs, LineDF_runs, label='Line DF', color='green', marker='x')
plt.scatter(runs, LineGI_runs, label='Line GI', color='red', marker='s')
plt.scatter(runs, LineAG_runs, label='Line AG', color='purple', marker='^')
plt.scatter(runs, LineBH_runs, label='Line BH', color='orange', marker='D')
plt.scatter(runs, LineCI_runs, label='Line CI', color='cyan', marker='P')

# Labels and title
plt.xlabel('Run Number')
plt.ylabel('Number of Users on Line')
plt.title('Scatter Plot of Users on Lines across Runs')
plt.legend()

# Layout and visualization
plt.tight_layout()
plt.show()
