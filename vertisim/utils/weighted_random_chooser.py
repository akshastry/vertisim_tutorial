import random
from typing import List, Any

def weighted_random_choose_exclude_element(elements_list: List,
                                           exclude_element: Any,
                                           probabilities: List,
                                           num_selection: int) -> List:
    
    # Create a new list with the desired probabilities for each element
    # Get the index of the exclude_element
    exclude_element_index = elements_list.index(exclude_element)
    # Evenly distribute the probability of the exclude_element to all other elements
    # The probability of the exclude_element is 0
    probabilities = [prob / (1 - probabilities[exclude_element_index]) if i != exclude_element_index else 0 for i, prob in enumerate(probabilities)]
    # Copy the elements_list and probabilities
    elements_list_ = elements_list.copy()
    probabilities_ = probabilities.copy()
    elements_list_.pop(exclude_element_index)
    probabilities_.pop(exclude_element_index)
    return [random.choices(elements_list_, weights=probabilities_)[0] for _ in range(num_selection)]

def random_choose_exclude_element(elements_list: List,
                                  exclude_element: Any,
                                  num_selection: int) -> List:
    # Choose num_selection elements from elements_list, excluding exclude_element
    return [random.choice([elem for elem in elements_list if elem != exclude_element]) for _ in range(num_selection)]
