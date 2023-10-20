# For nest NER dataset helper function.
# data: {'tokens': [], 'entities': [{'start': start, 'end': end, 'type': type}]}
def get_key_map(data_type):
    if data_type == 'ACE2004':
        key_map = {
            'PER': 'person',
            'ORG': 'organization',
            'LOC': 'location',
            'GPE': 'gpe',
            'FAC': 'facility',
            'WEA': 'weapon',
            'VEH': 'vehicle'
        }
    elif data_type == 'ACE2005':
        key_map = {
            'person': 'person',
            'organization': 'organization',
            'location': 'location',
            'gpe': 'gpe',
            'facility': 'facility',
            'weapon': 'weapon',
            'vehicle': 'vehicle'
        }
    return key_map

def get_keys(data_type):
    return set([v for k, v in get_key_map(data_type).items()])

def get_entity_type_desc(data_type):
    if data_type == 'ACE2004' or data_type == 'ACE2005':
        entity_type_desc = {
            'person': 'Person entities are limited to humans. A person may be a single individual or a group.',
            'organization': 'Organization entities are limited to corporations, agencies, and other groups of people defined by an established organizational structure.',
            'location': 'Location entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations',
            'gpe': 'GPE entities are geographical regions defined by political and/or social groups. A GPE entity subsumes and does not distinguish between a nation, its region, its government, or its people.',
            'facility': 'Facility entities are limited to buildings and other permanent man-made structures and real estate improvements',
            'vehicle': 'A vehicle entity is a physical device primarily designed to move an object from one location to another, by (for example) carrying, pulling, or pushing the transported object. Vehicle entities may or may not have their own power source.',
            'weapon': 'Weapon entities are limited to physical devices primarily used as instruments for physically harming or destroying animals (often humans), buildings, or other constructions.'
        }
    else:
        raise NotImplementedError('No this dataset.')
    return entity_type_desc