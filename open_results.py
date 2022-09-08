
import json

# Opening JSON file
f = open('300-25-8-1.json')


data = json.load(f)

agent_initial_pos =data['init_pos']
start_fire_node =data['init_pos']
sol=data['sol']

print(sol)

