from pdu.api import detonation_forward

def test_tritonal_kinetics():
    print("Testing V10.1 Miller Kinetics for Tritonal (80/20 TNT/Al)...")
    # Exp: D=6700, P=25.0, Q=7.50
    # V10.0 was P=17.0 (deficit -32%)
    res = detonation_forward(['TNT', 'Al'], [0.80, 0.20], 1.73, verbose=True)
    
    print(f"\nTritonal Results:")
    print(f"P_cj: {res.P_cj:.2f} GPa (Target 25.0, V10.0 was 17.0)")
    print(f"D_cj: {res.D:.0f} m/s (Target 6700)")
    print(f"Q: {res.Q:.2f} MJ/kg (Target 7.50, V10.0 was 10.23)")

if __name__ == "__main__":
    test_tritonal_kinetics()
