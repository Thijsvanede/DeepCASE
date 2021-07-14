from transitions import Machine

class RuleEngine(object):

    def __init__(self):
        pass

    def fit(self, rules):
        """Fit engine with rules"""
        # Initialise state machines
        self.machines = list()

        for category in rules:
            for directive in category['directives']:
                # Initialise states and transitions per rule
                states      = list()
                transitions = list()

                # Loop over subrules
                for rule in directive['rules']:
                    states.append(str(rule['depth']))
                    for event in rule['events']:
                        transitions.append({
                            'trigger': event,
                            'source' : str(rule['depth']),
                            'dest'   : str(rule['depth'] + 1),
                        })

                # Create FSM
                machine = Machine(
                    None,
                    states      = states,
                    transitions = transitions,
                    initial     = str(0.0),
                )

                # Add machine
                self.machines.append(machine)

        # Return self
        return self

    def predict(self, events):
        for index, event, timestamp in zip(events.index, events['threat_name'], events['ts_start']):

        pass


if __name__ == "__main__":
    import json
    from parse import load_mapping_events, load_mapping_ossim, map_rules

    with open('rules.json') as infile:
        rules = json.load(infile)

    # Load mapping of events
    mapping_events = load_mapping_events("../../../../data/fast.tmp.encoding.json")
    mapping_ossim  = load_mapping_ossim ('ossim_map_threat.csv')

    # Map rules
    rules = map_rules(
        rules          = rules,
        mapping_ossim  = mapping_ossim,
        mapping_events = mapping_events,
    )

    rsm = RuleEngine()
    rsm.fit(rules)

    # Load events
    import pandas as pd
    data = pd.read_csv('../../../../data/fast.tmp')[:500]

    # Preprocess
    data = data.sort_values(by='ts_start')
    from tqdm import tqdm
    # Predict per machine
    for machine, events in tqdm(data.groupby(['source', 'src_ip'])):
        # Apply rules to data
        rsm.predict(events)
