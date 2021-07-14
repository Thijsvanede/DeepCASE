import json
import yaml
from glob import glob

def load_rules():
    rules = dict()

    for path in glob('sigma/rules/*/*.yml') + glob('sigma/rules/*/*/*.yml'):
        # Get category
        category = path.split('/')[2]

        if category not in ['generic', 'network']: continue

        # Add category
        if category not in rules:
            rules[category] = {
                'category'  : category,
                'directives': list(),
            }

        with open(path, 'r') as infile:
            # Initialise rule
            rule = {
                'file'    : path,
                'subrules': list(),
                'category': category,
            }

            # Loop over documents in yaml file
            for doc in yaml.safe_load_all(infile):
                if 'title' in doc:
                    # Ensure no collisions
                    assert 'file'     not in doc
                    assert 'subrules' not in doc
                    # Update rule
                    rule.update(doc)

                    # Add subrules if applicable
                    if 'detection' in doc:
                        # Add subrules
                        rule['subrules'].append({
                            'detection': doc['detection']
                        })
                else:
                    if 'detection' in doc:
                        rule['subrules'].append(doc)

            # Append rule
            rules[category]['directives'].append(rule)

    return rules


def transform_rules(rules):
    result = list()

    detection_keys = set()

    event_index = 1

    for category, directives in rules.items():
        result.append({
            'category'  : category,
            'directives': list(),
        })

        for directive in directives.get('directives'):
            # Create new entry
            entry = {
                "title"   : directive.get('title'),
                "subtitle": directive.get('description'),
                "id"      : directive.get('id'),
                "rules"   : [
                    {
                        "depth"      : 0.0,
                        "name"       : directive.get('title').lower().replace(' ', '-'),
                        "reliability":   0,
                        "occurence"  :   1,
                        "timeout"    :  -1,
                        "from"       : "ANY",
                        "to"         : "ANY",
                        "event type" : {
                            str(event_index): "{}--{}".format(
                                directive.get('title').lower().replace(' ', '-'),
                                directive.get('id'),
                            )
                        }
                    }
                ],
            }

            event_index += 1

            # print("DIRECTIVE")

            for subrule in directive.get('subrules'):
                subrule = subrule['detection']
                # print(subrule)

            # Add entry
            result[-1]['directives'].append(entry)


    return result


if __name__ == "__main__":
    rules = load_rules()
    rules = transform_rules(rules)

    with open('rules.json', 'w') as outfile:
        json.dump(rules, outfile, indent='    ')
