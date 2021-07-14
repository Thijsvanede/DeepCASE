import argformat
import argparse
import csv
import json
import numpy  as np
import pandas as pd
from bs4         import BeautifulSoup
from collections import Counter
from tqdm        import tqdm

################################################################################
#                               Rule extraction                                #
################################################################################
def parse_directives_html(path):
    """Parse an HTML file containing directives and extract its rules.

        Parameters
        ----------
        path : string
            Path to HTML file containing directives.

        Returns
        -------
        rules : list of dict
            List of rules extracted from directives.
        """
    # Initialise result
    result = list()

    # Read file
    with open(path) as infile:
        html = infile.read()

    # Parse as HTML
    html = BeautifulSoup(html, 'html.parser')

    # Initialise rule id
    rule_id = 0

    # Extract content from HTML
    content = list(
        filter(
            lambda x: x != '\n',
            list(html.tbody.children))
        )[3].tbody

    # Loop over categories in content
    for category_html in filter(lambda x: x != '\n', content.children):
        # Get category
        category_html = category_html.div

        # Initialise category
        category = {
            "category"  : category_html.attrs.get('id'),
            "directives": list(),
        }

        # Check if category contains directives
        if category_html.tbody is not None:
            # Loop over all directives
            for directive_html in filter(lambda x: x != '\n', category_html.tbody.children):
                # Loop over rows in directive
                items = list(filter(lambda x: x != '\n', directive_html.children))

                # Case directive is title
                if len(items) == 3:
                    # Create new directive
                    directive = {
                        "title"   : items[2].a.text,
                        "subtitle": items[2].font.text,
                        "rules"   : [],
                    }

                # Case directive is rules
                elif len(items) == 1:
                    # Get rule ID
                    directive["id"] = int(
                        items[0].div.attrs.get('id').split('_')[1]
                    )

                    # Get subrules
                    rule     = list(filter(lambda x: x != '\n', items[0].tbody.children))[1]
                    subrules = list(filter(lambda x: x != '\n', rule    .tbody.children))[1:]

                    # Loop over all subrules
                    for subrule in subrules:
                        # Initialise subrule entry
                        subrule_entry = {
                            "depth"      : -1,
                            "name"       : None,
                            "reliability": None,
                            "timeout"    : None,
                            "occurence"  : None,
                            "from"       : None,
                            "to"         : None,
                            "data source": None,
                            "event type" : None,
                        }

                        # Extract data from subrules
                        for i, data in enumerate(filter(lambda x: x != '\n', subrule.children)):
                            # Add name and depth
                            if i == 0:
                                # Get style of item - Used to compute depth
                                style = dict(
                                    item.split(":")
                                    for item
                                    in data.attrs.get('style').split(';')
                                )
                                # Compute depth
                                depth = style.get("padding-left", "0px")[:-2]
                                depth = int(depth) / 19

                                subrule_entry['depth'] = depth
                                subrule_entry['name' ] = data.text.strip()
                            # Add reliability
                            if i == 1:
                                subrule_entry['reliability'] = int(data.text.strip())
                            # Add timeout
                            if i == 2:
                                timeout = data.text.strip()
                                if timeout == "None":
                                    timeout = -1
                                subrule_entry['timeout'] = int(timeout)
                            # Add occurence
                            if i == 3:
                                subrule_entry['occurence'] = int(data.text.strip())
                            # Add from
                            if i == 4:
                                subrule_entry['from'] = data.text.strip()
                            # Add to
                            if i == 5:
                                subrule_entry['to'] = data.text.strip()
                            # Add data source
                            if i == 6:
                                subrule_entry['data source'] = data.text.strip()
                            # Add event type
                            if i == 7:
                                # Initialise events
                                events = dict()
                                # Add all events
                                for event in data.find_all('a'):
                                    sid   = event.text.strip()
                                    event = event.attrs.get('title')

                                    if sid in events and events[sid] != event:
                                        raise ValueError("Double event")

                                    # Add event
                                    events[sid] = event

                                # Add events
                                subrule_entry['event type'] = events

                        # Add subrule
                        directive["rules"].append(subrule_entry)

                    # Add directive to category
                    category["directives"].append(directive)

                else:
                    raise ValueError("Unknown length")

        # Add category to result
        result.append(category)

    # Return result
    return result

################################################################################
#                                Event mapping                                 #
################################################################################

def get_events(rules):
    """Extract unique events from rules.

        Parameters
        ----------
        rules : list of dict
            Rules from which to extract unique events.

        Returns
        -------
        events : list()
            List of event names.
        """
    # Initialise result
    result = set()

    # Loop over rules
    for rule in rules:
        # Loop over events
        for event in rule.get('events', list()):
            result.add(event.get('event'))

    # Return as sorted list
    return list(sorted(result))


def load_mapping_events(path):
    """Load mapping of event name to ID."""
    with open(path) as infile:
        mapping = json.loads(infile.read())['threat_name']
    # Return mapping
    return {threat: i for i, threat in enumerate(mapping)}

def load_mapping_ossim(path):
    """Load mapping of OSSIM events to company events."""
    # Initialise result
    result = dict()

    # Threat mapping
    with open('ossim_map_threat.csv') as infile:
        # Create csv reader
        reader = csv.reader(infile)

        # Read files
        for entry in reader:
            original, *new = filter(None, entry)
            result[original] = new

    # Return result
    return result


def map_rules(rules, mapping_ossim, mapping_events):
    """Map rules from OSSIM to events."""
    # Loop over each rule
    for category in rules:
        # Loop over directives
        for directive in category["directives"]:
            # Loop over rules
            for rule in directive["rules"]:
                # Create events
                rule["events"] = set()

                # Loop over all events
                for sid, event in rule["event type"].items():
                    # Add corresponding events
                    rule["events"] |= set(mapping_ossim.get(event))

                # Create as mapping
                rule["events"] = {
                    mapping_events.get(event): event
                    for event in rule["events"]
                }

    # Return result
    return rules

################################################################################
#                                 Rule Engine                                  #
################################################################################


class RuleEngine(object):

    def __init__(self):
        """Match traffic against known rules."""
        # Initialise internal rules
        self.rules       = dict()
        self.event_rules = dict()
        self.states      = dict()

    ########################################################################
    #                       Fit and predict methods                        #
    ########################################################################

    def fit(self, rules):
        """Fit engine with rules.

            Parameters
            ----------
            rules : list of dict()
                List of rules containing 'events' that specify relevant events.

            Returns
            -------
            self : self
                Returns self.
            """
        # Store all rules
        self.all_rules = rules

        # Extract rules
        self.rules = dict()

        # Loop over each category
        for category in self.all_rules:
            # Loop over each directive
            for directive in category.get('directives'):
                # Get identifier
                identifier = directive.get('id')

                # Initialise rule
                rule = dict()

                # Get rules
                for subrule in directive.get('rules'):
                    # Get depth
                    depth = subrule['depth']

                    # Add starting depth rule
                    if depth not in rule:
                        # Add start rule
                        rule[depth] = {'start': subrule}

                    # Add closing depth rule
                    elif depth in rule and 'end' not in rule[depth]:
                        rule[depth]['end'] = subrule

                    # Unable to interpret rules
                    elif depth in rule and 'end' in rule[depth]:
                        raise ValueError(
                            "Unable to interpret rule hierarchy: {}"
                            .format(directive.get('rules'))
                        )

                # Add rule as identifier
                self.rules[identifier] = rule

        # Return self
        return self


    def predict(self, events, column='threat_name', sort='ts_start', groupby=['source', 'src_ip'], verbose=False):
        """Predict events based on rules.

            Parameters
            ----------
            events : pd.DataFrame
                DataFrame of events. Should contain following columns:
                - column    of sort parameter
                - column(s) of groupby parameter

            column : string, default='threat_name'
                Column from which to extract event.
                Default is events for Company dataset.

            sort : string, default='ts_start'
                Column to sort by.
                Default is time for Company dataset.

            groupby : list, default=['source', 'src_ip']
                List of columns to group by.
                Default is machine for Company dataset.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Array of matching rule IDs or -1 if no rule could be matched.
            """
        # Initialise result
        result = np.full(events.shape[0], -1)
        # Sort events - default by time
        events = events.sort_values(by=sort)
        # Group events - default by machine
        grouped_events = events.groupby(groupby)

        # Add verbose if necessary
        if verbose:
            grouped_events = tqdm(grouped_events, desc='Matching')

        # Loop over each group (machine) and get generated events
        for machine, events_ in grouped_events:
            # Initialise state of machine
            state = dict()

            # Loop over events
            for index, event in zip(events_.index, events_[column]):

                # Check each rule against event
                for identifier, rule in self.rules.items():

                    # If rule was previously triggered, also check deeper rules
                    if identifier in state:
                        # Get current depth
                        depth = state[identifier].get('depth')

                        # Check if rule is continued


                        # Check if rule is closed - TODO check
                        rule_close = rule.get(depth, {}).get('end', {})
                        if rule_close and event in rule_close.get('events', dict()):
                            # Add index
                            state[identifier]["indices"].append(index)
                            # Add counter
                            state[identifier]["counts"].update([event])

                            # Check if counter closes rule
                            occurence = rule_close['occurence']
                            total     = 0
                            for event_ in rule_close.get('events', dict()):
                                total += state[identifier]["counts"].get(event_, 0)

                            # Close rule if reached occurence limit
                            if total >= occurence:
                                # Get all indices of state
                                indices = state[identifier].get('indices', list())
                                # Update result
                                result[indices] = identifier
                                # Remove state
                                del state[identifier]
                                # Stop checking further
                                break

                    # Check rule start match
                    if event in rule[0].get('start', {}).get('events'):
                        # Add state
                        state[identifier] = {
                            "depth"  : 1,       # Set current depth of rule
                            "indices": [index], # Set indices related to alert
                            "counts" : Counter([event]), # Set counter
                        }

            # Set matched alerts
            for identifier, state_ in state.items():
                # Get all indices of state
                indices = state_.get('indices', list())
                # Update result
                result[indices] = identifier

        # Return result
        return result


    ########################################################################
    #                            Retrieve rules                            #
    ########################################################################

    def get_rules(self, event):
        """Get all matching rules based on event.

            Parameters
            ----------
            event : int
                ID of event for which to retrieve rules.

            Returns
            -------
            rules : list()
                List of relevant rules.
            """
        return [
            self.rules[id]                               # Get matching rule
            for id in self.event_rules.get(event, set()) # Get matching rule IDs
        ]




if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    parser = argparse.ArgumentParser(
        description     = "Compute #alerts when throttled",
        formatter_class = argformat.StructuredFormatter,
    )
    parser.add_argument('file', help=".csv file containing alerts")
    args = parser.parse_args()

    # with open('rules.json') as infile:
    #     rules = json.load(infile)
    #
    # print(len(rules))
    # total_directives = 0
    # for category in rules:
    #     directives = category.get('directives')
    #     total_directives += len(directives)
    # print(total_directives)
    # exit()

    ########################################################################
    #                         Load data and rules                          #
    ########################################################################
    # Extract rules
    rules = parse_directives_html('ossim/rules.html')

    # Save rules
    with open('ossim/rules.json', 'w') as outfile:
        json.dump(rules, outfile, indent='    ')

    # Load mapping of events
    mapping_events = load_mapping_events("{}.encoding.json".format(args.file))
    mapping_ossim  = load_mapping_ossim ('ossim_map_threat.csv')

    # Map rules
    rules = map_rules(
        rules          = rules,
        mapping_ossim  = mapping_ossim,
        mapping_events = mapping_events,
    )

    # Save mapped rules
    with open('ossim/rules_mapped.json', 'w') as outfile:
        json.dump(rules, outfile, indent='    ')

    # Load events
    data = pd.read_csv(args.file)

    ########################################################################
    #                          Create rule engine                          #
    ########################################################################

    # Create rule engine
    engine = RuleEngine()
    # Fit engine with rules
    engine.fit(rules)
    # Apply rules to data
    result = engine.predict(data, verbose=True)
