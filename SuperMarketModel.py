from queue import PriorityQueue
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


class Client:
    def __init__(self, arrival_time, path=None):
        self.arrival_time = arrival_time
        self.arrival_time_ff=0
        self.arrival_time_fm=0
        self.arrival_cash=0
        self.path = path if path is not None else []
    def next(self):
        if self.path:  #Check if there are still places in the path
            return self.path.pop(0)  # Returns and removes the first element
        return None  # If there are no more places, return None
    def set_arrival_ff(self,arrival_time):
        self.arrival_time_ff=arrival_time
    def set_arrival_fm(self,arrival_time):
        self.arrival_time_fm=arrival_time
    def set_arrival_cash(self,arrival_time):
        self.arrival_time_cash=arrival_time



class Measure:
     def __init__(self):
        self.total_delay=0 #Total delay
        self.start_delays=[] #Delays before going to desk/cash
        self.delays_ff = []  # Delays Fresh Food
        self.delays_fm = []  # Delays Fresh Meat
        self.delays_cash = []  # Delays Cash Registers
        self.queue_ff = []  # Fresh food queue sizes
        self.queue_fm = [] # Fresh meat queue sizes
        self.queue_cash=[] #Cash queue sizes

def arrival_supermarket(time, FES, queue_infinite):


    inter_arrival = random.expovariate(1.0 / ARRIVAL)
    FES.put((time + inter_arrival, "arrival"))
    path_choice = random.choices(
        [0, 1, 2, 3], 
        weights=[0.25,0.25, 0.25, 0.25], 
        k=1
    )[0]

    if path_choice == 0:  # -> Cash
        path = ["cash_register"]
    elif path_choice == 1:  # → Fresh Food -> Cash
        path = ["fresh_food", "cash_register"]
    elif path_choice == 2:  # → Fresh Meat -> Cash
        path = ["fresh_meat", "cash_register"]
    else:  # Fresh Food -> Fresh Meat -> Cash
        path = ["fresh_food","fresh_meat","cash_register"]
    
    client = Client(time,path)
    queue_infinite.append(client)
    service_time = random.expovariate(1.0 / SERVICE)
    FES.put((time + service_time, "departure_shopping"))
    
def departure_shopping(time, FES, queue_infinite, fresh_food_queue, fresh_meat_queue, cash_register_queue): 

    if len(queue_infinite) > 0:  
        client = queue_infinite.pop(0)
        data.total_delay+=(time - client.arrival_time)
        data.start_delays.append(time - client.arrival_time)
        next=client.next()
        if next=="cash_register":
            client.set_arrival_cash(time)
            FES.put((time,"arrival_cash"))
            cash_register_queue.append(client)
        elif next=="fresh_food":
            client.set_arrival_ff(time)
            FES.put((time,"arrival_fresh_food"))
            fresh_food_queue.append(client)
        elif next=="fresh_meat":
            client.set_arrival_fm(time)
            FES.put((time,"arrival_fresh_meat"))
            fresh_meat_queue.append(client)

        service_time = random.expovariate(1.0 / SERVICE)
        FES.put((time + service_time, "departure_shopping"))

        
def arrival_fresh_food(time, FES):
    global users_food
    

    users_food += 1
    if users_food <= NUM_SERVERS_FOOD:
        service_time = random.expovariate(1.0 / FRESH_FOOD_SERVICE)
        FES.put((time + service_time, "departure_fresh_food"))

def departure_fresh_food(time, FES, fresh_food_queue, fresh_meat_queue, cash_register_queue):
    global users_food

    if len(fresh_food_queue) > 0:
        client = fresh_food_queue.pop(0)
        data.total_delay+=(time - client.arrival_time_ff)
        data.delays_ff.append(time - client.arrival_time_ff)
        users_food-=1

        next=client.next()
        if next=="cash_register":
            client.set_arrival_cash(time)
            FES.put((time,"arrival_cash"))
            cash_register_queue.append(client)
        elif next=="fresh_meat":
            client.set_arrival_fm(time)
            FES.put((time,"arrival_fresh_meat"))
            fresh_meat_queue.append(client)
        if users_food>0:
            service_time = random.expovariate(1.0 / FRESH_FOOD_SERVICE)
            FES.put((time + service_time, "departure_fresh_food"))

def arrival_fresh_meat(time, FES):
    global users_meat
    users_meat += 1
    if users_meat <= NUM_SERVERS_MEAT:
        service_time = random.expovariate(1.0 / FRESH_MEAT_SERVICE)
        FES.put((time + service_time, "departure_fresh_meat"))

def departure_fresh_meat(time, FES, fresh_meat_queue, cash_register_queue):
    global users_meat

    if len(fresh_meat_queue) > 0:
        client = fresh_meat_queue.pop(0)
        data.total_delay+=(time - client.arrival_time_fm)
        data.delays_fm.append(time - client.arrival_time_fm)
        users_meat-=1
        next=client.next()
        if next=="cash_register":
            client.set_arrival_cash(time)
            FES.put((time,"arrival_cash"))
            cash_register_queue.append(client)
        if users_meat>0:
            service_time = random.expovariate(1.0 / FRESH_MEAT_SERVICE)
            FES.put((time + service_time, "departure_fresh_meat"))

def arrival_cash(time, FES):
    global users_cash
    
    if len(cash_register_queue) > 0:
        users_cash += 1
        if users_cash <= NUM_SERVERS_CASH:
            service_time = random.expovariate(1.0 / CASH_REGISTER_SERVICE)
            FES.put((time + service_time, "departure_supermarket"))

def departure_supermarket(time,FES,cash_register_queue):
    global users_cash

    if len(cash_register_queue)>0:
        client = cash_register_queue.pop(0)
        data.total_delay+=(time - client.arrival_time_cash)
        data.delays_cash.append(time - client.arrival_time_cash)
        users_cash-=1
        if users_cash>0:
            service_time = random.expovariate(1.0 / CASH_REGISTER_SERVICE)
            FES.put((time + service_time, "departure_supermarket"))

# Input parameters
NUM_SERVERS_FOOD = 2
NUM_SERVERS_MEAT = 2
NUM_SERVERS_CASH = 3
ARRIVAL = 1
SERVICE = 2
FRESH_FOOD_SERVICE = 4
FRESH_MEAT_SERVICE = 5
CASH_REGISTER_SERVICE = 6
# Parameters for the runs and confidence intervals
NUM_RUNS = 30
CONF_LEVEL = 0.9

# Lists for storing results of each run
total_delay_runs=[]
start_delays_runs = []
delays_ff_runs = []
delays_fm_runs = []
delays_cash_runs = []
queue_ff_runs = []
queue_fm_runs = []
queue_cash_runs = []

for i in range(NUM_RUNS):

    data = Measure()
    time = 0
    FES = PriorityQueue()

    queue_infinite = []  
    fresh_food_queue = []  
    fresh_meat_queue = []  
    cash_register_queue = []  

    users_food = 0
    users_meat = 0
    users_cash = 0  

    FES.put((0, "arrival"))

    event_count = 0
    max_time=1000

    while time < max_time:
        time, event_type = FES.get()
        if event_type == "arrival":
            arrival_supermarket(time,FES,queue_infinite)

        elif event_type == "departure_shopping":
            departure_shopping(time,FES,queue_infinite,fresh_food_queue,fresh_meat_queue,cash_register_queue)

        elif event_type == "arrival_fresh_food":
            arrival_fresh_food(time,FES)

        elif event_type == "departure_fresh_food":
            departure_fresh_food(time,FES,fresh_food_queue,fresh_meat_queue,cash_register_queue)

        elif event_type == "arrival_fresh_meat":
            arrival_fresh_meat(time,FES)

        elif event_type == "departure_fresh_meat":
            departure_fresh_meat(time,FES,fresh_meat_queue,cash_register_queue)

        elif event_type == "arrival_cash":
            arrival_cash(time,FES)

        elif event_type == "departure_supermarket":
            departure_supermarket(time,FES,cash_register_queue)

        data.queue_ff.append(len(fresh_food_queue))
        data.queue_fm.append(len(fresh_meat_queue))
        data.queue_cash.append(len(cash_register_queue))
        event_count+=1

   # Calculation of statistics
    if data.start_delays:
        average_start_delay = np.mean(data.start_delays)
        delay_start_std = np.std(data.start_delays)
    else:
        average_start_delay = 0
        delay_start_std = 0

    if data.delays_ff:
        average_delay_ff = np.mean(data.delays_ff)
        delay_ff_std = np.std(data.delays_ff)
    else:
        average_delay_ff  = 0
        delay_ff_std = 0

    if data.delays_fm:
        average_delay_fm = np.mean(data.delays_fm)
        delay_fm_std = np.std(data.delays_fm)
    else:
        average_delay_fm  = 0
        delay_fm_std = 0

    if data.delays_cash:
        average_delay_cash = np.mean(data.delays_cash)
        delay_cash_std = np.std(data.delays_cash)
    else:
        average_delay_cash  = 0
        delay_cash_std = 0


    average_queue_ff = np.mean(data.queue_ff) if data.queue_ff else 0
    average_queue_fm = np.mean(data.queue_fm) if data.queue_fm else 0
    average_queue_cash = np.mean(data.queue_cash) if data.queue_cash else 0

    print(f"Run Number: {i}")
    print(f"Total delay: {data.total_delay}")
    print(f"Mean Start delay: {average_start_delay}")
    print(f"Standard deviation of Start delay: {delay_start_std}")
    print(f"Mean Fresh Food delay: {average_delay_ff}")
    print(f"Standard deviation of Fresh food delay: {delay_ff_std}")
    print(f"Mean Fresh Meat delay: {average_delay_fm}")
    print(f"Standard deviation of Fresh meat delay: {delay_fm_std}")
    print(f"Mean Cash Register delay: {average_delay_cash}")
    print(f"Standard deviation of Cash register delay: {delay_cash_std}")
    print(f"Mean Fresh Food queue size : {average_queue_ff}")
    print(f"Mean Fresh Meat queue size : {average_queue_fm}")
    print(f"Mean Cash Registers queue size : {average_queue_cash}")


    print(f"Total events processed: {event_count}")
    print()
    total_delay_runs.append(data.total_delay)
    start_delays_runs.append(np.mean(data.start_delays))
    delays_ff_runs.append(np.mean(data.delays_ff))
    delays_fm_runs.append(np.mean(data.delays_fm))
    delays_cash_runs.append(np.mean(data.delays_cash))
    queue_ff_runs.append(np.mean(data.queue_ff))
    queue_fm_runs.append(np.mean(data.queue_fm))
    queue_cash_runs.append(np.mean(data.queue_cash))

    #Since I'm making lots of runs I'm not plotting all the histograms
    '''
    # Histogram of initial delays
    plt.hist(data.start_delays, bins='auto')
    plt.title("Histogram of Start Delays")
    plt.xlabel("Delay (time units)")
    plt.ylabel("Frequency")
    plt.show()

    # Histogram of delays at the Fresh Food queue
    plt.hist(data.delays_ff, bins='auto')
    plt.title("Histogram of Fresh Food Delays")
    plt.xlabel("Delay (time units)")
    plt.ylabel("Frequency")
    plt.show()

    # Histogram of delays at the Fresh Meat queue
    plt.hist(data.delays_fm, bins='auto')
    plt.title("Histogram of Fresh Meat Delays")
    plt.xlabel("Delay (time units)")
    plt.ylabel("Frequency")
    plt.show()

    # Histogram of Delays at the Cash Registers Queue
    plt.hist(data.delays_cash, bins='auto')
    plt.title("Histogram of Cash Register Delays")
    plt.xlabel("Delay (time units)")
    plt.ylabel("Frequency")
    plt.show()



    plt.hist(data.queue_ff, bins='auto')
    plt.title("Histogram of Fresh Food Queue Sizes")
    plt.xlabel("Queue Size")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(data.queue_fm, bins='auto')
    plt.title("Histogram of Fresh Meat Queue Sizes")
    plt.xlabel("Queue Size")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(data.queue_cash, bins='auto')
    plt.title("Histogram of Cash Registers Queue Sizes")
    plt.xlabel("Queue Size")
    plt.ylabel("Frequency")
    plt.show()
    '''

#confidence intervals
def confidence_interval(data, confidence=CONF_LEVEL):
    mean_val = np.mean(data)
    sem = st.sem(data)  # Standard error of the mean
    interval = st.t.interval(confidence, len(data)-1, loc=mean_val, scale=sem)
    return mean_val, interval

avg_total_delay, ci_total_delay = confidence_interval(total_delay_runs)
avg_start_delay, ci_start_delay = confidence_interval(start_delays_runs)
avg_delay_ff, ci_delay_ff = confidence_interval(delays_ff_runs)
avg_delay_fm, ci_delay_fm = confidence_interval(delays_fm_runs)
avg_delay_cash, ci_delay_cash = confidence_interval(delays_cash_runs)
avg_queue_ff, ci_queue_ff = confidence_interval(queue_ff_runs)
avg_queue_fm, ci_queue_fm = confidence_interval(queue_fm_runs)
avg_queue_cash, ci_queue_cash = confidence_interval(queue_cash_runs)

# Output of results with confidence intervals
print("Confidence intervals for metrics:")
print(f"Mean Total delay: {avg_total_delay} with CI at {CONF_LEVEL*100}% : {ci_total_delay}")
print(f"Mean Start delay: {avg_start_delay} with CI at {CONF_LEVEL*100}% : {ci_start_delay}")
print(f"Mean Fresh Food delay: {avg_delay_ff} with CI at {CONF_LEVEL*100}% : {ci_delay_ff}")
print(f"Mean Fresh Meat delay: {avg_delay_fm} with CI at {CONF_LEVEL*100}% : {ci_delay_fm}")
print(f"Mean Cash Register delay: {avg_delay_cash} with CI at {CONF_LEVEL*100}% : {ci_delay_cash}")
print(f"Mean Fresh Food queue size: {avg_queue_ff} with CI at {CONF_LEVEL*100}% : {ci_queue_ff}")
print(f"Mean Fresh Meat queue size: {avg_queue_fm} with CI at {CONF_LEVEL*100}% : {ci_queue_fm}")
print(f"Mean Cash Registers queue size: {avg_queue_cash} with CI at {CONF_LEVEL*100}% : {ci_queue_cash}")