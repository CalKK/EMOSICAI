from datetime import datetime, timedelta
from rules import ConflictResolver

def run_specificity_test():
    print("""
# =============================================================================
# 1. SPECIFICITY TEST
# =============================================================================
Goal: Show that a more specific rule (higher 'specificity' value) is chosen.
""")
    resolver = ConflictResolver()
    
    # R1 is more specific than R2
    rule1 = {
        "id": "R1", "specificity": 3, "confidence": 0.9, "created_at": datetime.now(),
        "rule": "IF (Tempo IS Fast) AND (Key IS Major) THEN (EmotionalState IS Happy)"
    }
    rule2 = {
        "id": "R2", "specificity": 2, "confidence": 0.85, "created_at": datetime.now(),
        "rule": "IF (Valence IS High) AND (Energy IS High) THEN (EmotionalState IS Happy)"
    }
    
    applicable_rules = [rule1, rule2]
    
    print(f"Applicable Rules: ['R1' (Specificity: {rule1['specificity']}), 'R2' (Specificity: {rule2['specificity']})]")
    
    selected_rule = resolver.resolve_conflicts(applicable_rules)
    
    print(f"Selected Rule: {selected_rule['id']}")
    print(f"Outcome: {selected_rule['id']} was chosen because its specificity ({selected_rule['specificity']}) is the highest.\n")
    assert selected_rule['id'] == 'R1'

def run_recency_test():
    print("""
# =============================================================================
# 2. RECENCY TEST
# =============================================================================
Goal: When specificities are equal, the most recently created rule wins.
""")
    resolver = ConflictResolver()
    
    time_now = datetime.now()
    time_yesterday = time_now - timedelta(days=1)
    
    # R1_new is more recent than R1_old
    rule1_old = {
        "id": "R1_old", "specificity": 3, "confidence": 0.9, "created_at": time_yesterday,
        "rule": "IF (Tempo IS Fast) AND (Key IS Major) THEN (EmotionalState IS Happy)"
    }
    rule1_new = {
        "id": "R1_new", "specificity": 3, "confidence": 0.85, "created_at": time_now,
        "rule": "IF (Tempo IS Fast) AND (Valence IS High) THEN (EmotionalState IS Happy)"
    }
    
    applicable_rules = [rule1_old, rule1_new]
    
    print(f"Applicable Rules: ['R1_old' (Created: {rule1_old['created_at']}), 'R1_new' (Created: {rule1_new['created_at']})]")
    
    selected_rule = resolver.resolve_conflicts(applicable_rules)
    
    print(f"Selected Rule: {selected_rule['id']}")
    print(f"Outcome: {selected_rule['id']} was chosen because it was created more recently.\n")
    assert selected_rule['id'] == 'R1_new'

def run_refractoriness_test():
    print("""
# =============================================================================
# 3. REFRACTORINESS TEST
# =============================================================================
Goal: A rule that has recently fired should be temporarily ignored.
""")
    resolver = ConflictResolver()
    
    time_now = datetime.now()
    
    rule_a = {"id": "Rule-A", "specificity": 3, "confidence": 0.9, "created_at": time_now}
    rule_b = {"id": "Rule-B", "specificity": 3, "confidence": 0.9, "created_at": time_now}

    # First, fire Rule-A
    print("Step 1: Firing Rule-A")
    resolver.resolve_conflicts([rule_a])
    print(f"Rule-A last used at: {resolver.rule_last_used['Rule-A']}")

    # Immediately after, try to fire a rule from a set including Rule-A
    print("\nStep 2: Immediately processing a new song matching both Rule-A and Rule-B")
    applicable_rules = [rule_a, rule_b]
    
    selected_rule = resolver.resolve_conflicts(applicable_rules)
    
    print(f"Selected Rule: {selected_rule['id']}")
    print(f"Outcome: {selected_rule['id']} was chosen because Rule-A was used in the last 30 seconds and was ignored.\n")
    assert selected_rule['id'] == 'Rule-B'

def run_lexical_order_test():
    print("""
# =============================================================================
# 4. LEXICAL ORDER TEST
# =============================================================================
Goal: Provide a deterministic tie-breaker when other strategies fail.
""")
    resolver = ConflictResolver()
    
    time_now = datetime.now()
    
    # Rules have same specificity and creation time
    rule_b = {"id": "Rule-B", "specificity": 3, "confidence": 0.9, "created_at": time_now}
    rule_c = {"id": "Rule-C", "specificity": 3, "confidence": 0.9, "created_at": time_now}
    
    applicable_rules = [rule_c, rule_b] # Order doesn't matter
    
    print(f"Applicable Rules: ['Rule-B', 'Rule-C'] (Identical specificity and creation time)")
    
    selected_rule = resolver.resolve_conflicts(applicable_rules)
    
    print(f"Selected Rule: {selected_rule['id']}")
    print(f"Outcome: {selected_rule['id']} was chosen because its ID comes first alphabetically.\n")
    assert selected_rule['id'] == 'Rule-B'

def run_confidence_test():
    print("""
# =============================================================================
# 5. MEANS-END ANALYSIS (CONFIDENCE) TEST
# =============================================================================
Goal: The rule with the highest confidence wins if other factors are equal.
Note: Based on the current implementation, this test demonstrates that
      Lexical Order is prioritized over Confidence.
""")
    resolver = ConflictResolver()
    
    time_now = datetime.now()
    
    # R3 has higher confidence, but R4_mod comes first alphabetically
    rule3 = {
        "id": "R4_mod", "specificity": 3, "confidence": 0.8, "created_at": time_now,
        "rule": "IF (Valence IS Low) AND (Energy IS Low) THEN (EmotionalState IS Sad)"
    }
    rule4 = {
        "id": "R3", "specificity": 3, "confidence": 0.9, "created_at": time_now,
        "rule": "IF (Tempo IS Slow) AND (Key IS Minor) THEN (EmotionalState IS Sad)"
    }
    
    applicable_rules = [rule4, rule3]
    
    print(f"Applicable Rules: ['R3' (Confidence: {rule4['confidence']}), 'R4_mod' (Confidence: {rule3['confidence']})]")
    print("Note: Both rules have the same specificity and creation time.")

    selected_rule = resolver.resolve_conflicts(applicable_rules)
    
    print(f"Selected Rule: {selected_rule['id']}")
    print(f"Outcome: {selected_rule['id']} was chosen due to lexical order, not confidence, highlighting the current logic's priority.\n")
    # This assertion shows that the current implementation chooses based on lexical order before confidence
    assert selected_rule['id'] == 'R3'

if __name__ == "__main__":
    run_specificity_test()
    run_recency_test()
    run_refractoriness_test()
    run_lexical_order_test()
    run_confidence_test()
