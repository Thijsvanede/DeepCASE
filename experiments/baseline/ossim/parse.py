from bs4 import BeautifulSoup
import csv
import json

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
    with open(path) as infile:
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
