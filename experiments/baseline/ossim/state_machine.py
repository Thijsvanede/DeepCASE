from copy        import deepcopy
from collections import Counter

class RuleSequence(object):

    def __init__(self, rules):
        # Create a state for each rule
        states = self.create_states(rules)
        # Link states together
        self.states = self.link_states(rules, states)
        # Set initial and current state
        self.initial = self.states[0]
        self.current = self.states[0]

    ########################################################################
    #                           Get transitions                            #
    ########################################################################

    @property
    def transitions(self):
        return set(self.current.transitions.keys())

    ########################################################################
    #                            Create states                             #
    ########################################################################


    def create_states(self, rules):
        """Create RuleStates from rules.

            Parameters
            ----------
            rules : list()
                List of rules from which to create states.

            Returns
            -------
            states : list()
                List of RuleState objects representing all states.
            """
        # Create states from subrules
        states = list()
        # Loop over subrules
        for rule in rules:
            # Create state for subrule
            state = RuleState(
                identifier  = rule['depth'],
                description = rule['name'],
                reliability = rule['reliability'],
                timeout     = rule['timeout'] if rule['timeout'] >= 0 else float('inf'),
                occurence   = rule['occurence'],
                final       = False,
            )

            # Append to states
            states.append(state)

        # Return states
        return states


    def link_states(self, rules, states):
        """Link all states using the given transition events.

            Parameters
            ----------
            rules : list()
                List of rules from which to create states.

            states : list()
                List of RuleState objects representing all states.

            Returns
            -------
            state : RuleState()
                Initial state of linked RuleState objects.
            """
        # In case of a single state, create transition to closed state
        if len(rules) == 1:
            # Create closed state
            rule  = rules[0]
            state = states[0]
            state_closed = RuleState(
                identifier  = "{}_closed" .format(state.identifier ),
                description = "Closed: {}".format(state.description),
                reliability = state.reliability,
                timeout     = -1,
                occurence   = -1,
                final       = True,
            )

            # Add transition
            for event in rule.get('events', list()):
                state.add_transition(event, state_closed)

        # Otherwise, do not create closed state
        else:
            # Store previous objects
            previous = dict()

            # Loop over all rules and next states
            for current_rule, current_state in zip(rules, states):
                # Check if previous state exists
                if previous:
                    # Get depth
                    depth = current_rule['depth']

                    # Case of next depth
                    if depth not in previous:

                        # Add links from previous to current state
                        for event in previous[depth - 1]['rule']['events']:
                            # Add link
                            previous[depth - 1]['state'].add_transition(
                                event = event,
                                state = current_state
                            )

                    # Case of closing state
                    else:
                        # Ignore if occurence is very large - ignore exit
                        # TODO: IMPORTANT - This has to be fixed when used in
                        # practice, but is fine for experiments as rules are not
                        # triggered mroe than 10K times
                        if current_rule['occurence'] < 10000:

                            # Create closed state
                            state_closed = RuleState(
                                identifier  = "{}_closed" .format(current_state.identifier ),
                                description = "Closed: {}".format(current_state.description),
                                reliability = current_state.reliability,
                                timeout     = -1,
                                occurence   = -1,
                                final       = True,
                            )

                            # Add events to closing state
                            for event in current_rule['events']:
                                # Add exit transitions
                                previous[depth]['state'].add_transition(
                                    event  = event,
                                    state  = state_closed,
                                    strict = 'original', # TODO: IMPORTANT - Currently a state is always closed on first instance
                                )

                # Set previous rule and state
                previous[current_rule['depth']] = {
                    'rule' : current_rule,
                    'state': current_state,
                }

        # Return initial state of linked states
        return states

    ########################################################################
    #                             Transitions                              #
    ########################################################################

    def reset(self):
        """Reset state sequence"""
        # Reset state
        self.current            = self.initial
        self.current.counts     = Counter()
        self.current.first_seen = None

    def transition(self, event, timestamp):
        """Take transition with event and its timestamp.

            Parameters
            ----------
            event : int
                ID of event for which to take transition.

            timestamp : float
                Timestamp of event, used to compute timeout.

            Returns
            -------
            success : boolean
                True if event matched possible transition without timing out,
                False otherwise.

            state : RuleState
                State after transition.
            """
        # Get current state
        counts     = self.current.counts
        first_seen = self.current.first_seen

        # Perform transition on current state
        success, self.current = self.current.transition(event, timestamp)

        # If successful, transfer counts
        if success:
            self.current.counts     = counts
            self.current.first_seen = first_seen

        # Return result
        return success, self.current



    ########################################################################
    #                          Auxiliary methods                           #
    ########################################################################


    def __str__(self):
        """Returns string representation of RuleSequence."""
        result = "StateSequence:\n┣ {}\n".format(self.initial.identifier)

        # Create state stack
        stack = [self.initial]
        depth = 0

        # Loop over states
        while stack:
            # Pop last state from the stack
            state = stack.pop()

            # Get transitions
            transitions = state.transition_summary()
            # Check if we have transitions
            if transitions:

                # Print current state
                # Add depth
                depth += 1

                # Get all transitions
                for next_state, transition in transitions.items():
                    # Append result
                    result += "┣{} {} ← {}\n".format('━'*2*depth, next_state.identifier, transition)
                    # Add to stack
                    stack.append(next_state)

        # Return result
        return result



class RuleState(object):

    def __init__(self, identifier, description, reliability, timeout, occurence, first_seen=None, final=False):
        """State of a rule.

            Parameters
            ----------
            identifier : Object
                Unique identifier for rule state.

            description : string
                String describing rule state

            reliability : int
                Indication of reliability of rule state.

            timeout : float
                Maximum number of seconds before timing out.

            occurence : int
                Minimum number of occurences required for transition.

            first_seen : float, default=None
                If given, set first_seen

            final : boolean, default=False
                If True, indicate this is a final state.
            """
        # Initialise state
        self.identifier  = identifier
        self.description = description
        self.reliability = reliability
        self.timeout     = timeout
        self.occurence   = occurence

        # Initialise time for timeout
        self.first_seen  = first_seen

        # Initialise states
        self.final       = final
        self.counts      = Counter()
        self.transitions = dict()

        # Add timeout state if necessary
        self.add_timeout_state()


    def add_transition(self, event, state, strict=True):
        """Add transition to next state from event."""
        # Always keep original value
        if strict == 'original' and event in self.transitions:
            return

        # Perform checks if strict
        if strict and event in self.transitions and self.transitions[event] != state:
            raise ValueError(
                "Double transition found: {} -> {} and {}"
                .format(event, self.transitions[event], state)
            )

        # Add transition
        self.transitions[event] = state


    def add_timeout_state(self):
        """Automatic method for adding timeout state with transition.

            Returns
            -------
            self : self
                Returns self.
            """
        if self.timeout >= 0 and self.timeout != float('inf'):
            # Create timeout state
            timeout_state = RuleState(
                identifier  = "{}_timeout" .format(self.identifier ),
                description = "Timeout: {}".format(self.description),
                reliability = self.reliability,
                timeout     = -1,
                occurence   = -1,
                first_seen  = self.first_seen,
                final       = True,
            )

            # Add transition
            self.add_transition('timeout', timeout_state)

        # Return self
        return self


    def transition(self, event, timestamp):
        """Take transition with event and its timestamp.

            Parameters
            ----------
            event : int
                ID of event for which to take transition.

            timestamp : float
                Timestamp of event, used to compute timeout.

            Returns
            -------
            success : boolean
                True if event matched possible transition without timing out,
                False otherwise.

            state : RuleState
                State after transition.
            """
        # Check for timeout
        if self.first_seen is not None and\
           timestamp - self.first_seen > self.timeout:
            # Return failed and timeout state
            return False, self.transitions['timeout']

        # Check if transition is possible, i.e. matches rule
        if event in self.transitions:
            # Set initial timestamp
            if self.first_seen is None: self.first_seen = timestamp
            # Add to counter
            self.counts.update([event])
            # Check if counts is enough
            if sum(self.counts.values()) >= self.occurence:
                # Return success and next state
                return True, self.transitions[event]
            else:
                return True, self

        # Otherwise return failed and current state
        return False, self


    def transition_summary(self):
        """Return a transition summary.

            Returns
            -------
            result : dict()
                Dictionary of possible next states with each transition.
            """
        # Initialise result
        result = dict()

        # Add all transitions
        for transition, state in self.transitions.items():
            if state not in result:
                result[state] = list()
            result[state].append(transition)

        # Return summary
        return result


    def __str__(self):
        """String representation of state."""
        # Get transitions
        transitions = ""
        for event, state in self.transitions.items():
            # Add transition
            transitions += "\t{} → {}\n".format(
                event,
                "None" if state is None else state.identifier
            )

        return """RuleState(
    identifier : {},
    description: {},
    reliability: {},
    timeout    : {},
    occurence  : {},
    final      : {},
)
Transitions:
{}""".format(
    self.identifier ,
    self.description,
    self.reliability,
    self.timeout    ,
    self.occurence  ,
    self.final      ,
    transitions if transitions else "\tNone"
)


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

    rsm = RuleStateMachine()
    rsm.fit(rules)

    # Load events
    import pandas as pd
    data = pd.read_csv('../../../../data/fast.tmp', nrows=500)

    # Preprocess
    data = data.sort_values(by='ts_start')
    from tqdm import tqdm
    # Predict per machine
    for machine, events in tqdm(data.groupby(['source', 'src_ip'])):
        # Apply rules to data
        rsm.predict(events['threat_name'], events['ts_start'].values)
