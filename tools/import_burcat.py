
import xml.etree.ElementTree as ET
import json
import sys
# import jax.numpy as jnp # Not needed for parsing

def parse_burcat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Target species mapping: {XML_Formula: PDU_Name}
    targets = {
        'ALO': 'AlO',
        'AL2O': 'Al2O',
        'ALO2': 'AlO2',
        'ALOH': 'AlOH',
        'AL2O3(L)': 'Al2O3_L_Burcat' 
    }
    
    extracted = {}
    
    for specie in root.findall('specie'):
        for phase in specie.findall('phase'):
            formula = phase.find('formula').text.strip()
            
            # Match strictly
            if formula in targets:
                pdu_name = targets[formula]
                print(f"Found {pdu_name} ({formula})")
                
                # Extract molecular weight
                mw = float(phase.find('molecular_weight').text)
                
                # Extract coefficients
                coeffs_node = phase.find('coefficients')
                
                # High Temp (1000-Tmax)
                range_high = coeffs_node.find('range_1000_to_Tmax')
                c_high = [
                    float(range_high.find(f'coef[@name="a{i}"]').text)
                    for i in range(1, 8)
                ]
                
                # Low Temp (Tmin-1000)
                range_low = coeffs_node.find('range_Tmin_to_1000')
                c_low = [
                    float(range_low.find(f'coef[@name="a{i}"]').text)
                    for i in range(1, 8)
                ]
                
                # Determine phase type
                state_chk = phase.find('phase').text.strip()
                phase_type = 'gas'
                if state_chk == 'S': phase_type = 'solid'
                if state_chk == 'L': phase_type = 'liquid'
                
                extracted[pdu_name] = {
                    "name": pdu_name,
                    "molecular_weight": mw,
                    "phase": phase_type,
                    "coeffs_high": c_high,
                    "coeffs_low": c_low,
                    #"source": "Burcat XML"
                }

    return extracted

def update_products_json(extracted_data, json_path):
    with open(json_path, 'r') as f:
        db = json.load(f)
        
    # Update species
    for name, data in extracted_data.items():
        db['species'][name] = data
        print(f"Updated/Added {name} to products.json")
        
    # Save
    with open(json_path, 'w') as f:
        json.dump(db, f, indent=2)

if __name__ == "__main__":
    data = parse_burcat_xml("pdu/data/BURCAT_THR.xml")
    update_products_json(data, "pdu/data/products.json")
