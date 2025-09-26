import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        
    def get_value(self):
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 11  # Ace value will be adjusted in Hand class
        else:
            return int(self.value)
    
    def __repr__(self):
        return f"{self.value}{self.suit}"

class Deck:
    def __init__(self, num_decks=8):
        self.num_decks = num_decks
        self.cards = []
        self.dealt_cards = []
        self.reset()
        
    def reset(self):
        self.cards = []
        self.dealt_cards = []
        suits = ['H', 'D', 'C', 'S']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        for _ in range(self.num_decks):
            for suit in suits:
                for value in values:
                    self.cards.append(Card(suit, value))
        
        random.shuffle(self.cards)
    
    def deal(self):
        if not self.cards:
            return None
        
        card = self.cards.pop(0)
        self.dealt_cards.append(card)
        return card
    
    def cards_remaining(self):
        return len(self.cards)
    
    def needs_shuffle(self):
        # Reshuffle after 300 cards have been dealt
        return len(self.dealt_cards) >= 300
    
    def get_card_count(self):
        # Returns dictionary with count of each card value left in the deck
        count = defaultdict(int)
        for card in self.cards:
            count[card.value] += 1
        return count

class Hand:
    def __init__(self):
        self.cards = []
        self.value = 0
        self.aces = 0
        self.is_soft = False
        self.is_blackjack = False
        self.can_split = False
        self.is_splitted = False
        
    def add_card(self, card):
        self.cards.append(card)
        self._calculate_value()
        self._check_split()
        
    def _calculate_value(self):
        self.value = 0
        self.aces = 0
        
        for card in self.cards:
            if card.value == 'A':
                self.aces += 1
            self.value += card.get_value()
        
        # Adjust for aces
        self.is_soft = False
        if self.aces > 0 and self.value > 21:
            self.is_soft = True
            for _ in range(self.aces):
                if self.value > 21:
                    self.value -= 10
                    self.aces -= 1
                else:
                    break
        
        # Check for blackjack
        if len(self.cards) == 2 and self.value == 21 and not self.is_splitted:
            self.is_blackjack = True
    
    def _check_split(self):
        if len(self.cards) == 2 and self.cards[0].value == self.cards[1].value:
            self.can_split = True
        else:
            self.can_split = False
    
    def is_busted(self):
        return self.value > 21
    
    def is_pair(self):
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value
    
    def get_first_card(self):
        if self.cards:
            return self.cards[0]
        return None
    
    def __repr__(self):
        hand_str = ", ".join(str(card) for card in self.cards)
        return f"{hand_str} ({self.value})"

class CardCounter:
    def __init__(self, strategy='hi-lo'):
        self.strategy = strategy
        self.running_count = 0
        self.true_count = 0
        self.decks_remaining = 8.0
        
        # Define count values for different strategies
        if strategy == 'hi-lo':
            self.count_values = {
                '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
                '7': 0, '8': 0, '9': 0,
                '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
            }
        elif strategy == 'hi-opt-1':
            self.count_values = {
                '2': 0, '3': 1, '4': 1, '5': 1, '6': 1,
                '7': 0, '8': 0, '9': 0,
                '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': 0
            }
        elif strategy == 'hi-opt-2':
            self.count_values = {
                '2': 1, '3': 1, '4': 2, '5': 2, '6': 1,
                '7': 1, '8': 0, '9': 0,
                '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': 0
            }
        elif strategy == 'omega-2':
            self.count_values = {
                '2': 1, '3': 1, '4': 2, '5': 2, '6': 2,
                '7': 1, '8': 0, '9': -1,
                '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': 0
            }
        else:  # Default to Hi-Lo
            self.count_values = {
                '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
                '7': 0, '8': 0, '9': 0,
                '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
            }
    
    def update_count(self, card):
        # DISABLED: No longer updating count for basic strategy only
        pass
    
    def update_decks_remaining(self, cards_remaining):
        self.decks_remaining = cards_remaining / 52
    
    def reset(self):
        self.running_count = 0
        self.true_count = 0
        self.decks_remaining = 8.0
    
    def get_bet_amount(self, min_bet=50, max_bet=5000):
        """
        MODIFIED: Returns flat bet amount (ignoring count)
        """
        return min_bet  # Always bet the minimum amount
    
    def __repr__(self):
        return f"Running Count: {self.running_count}, True Count: {self.true_count:.2f}, Decks Remaining: {self.decks_remaining:.2f}"

class BlackjackEnv:
    def __init__(self, num_decks=8, min_bet=50, max_bet=5000):
        self.deck = Deck(num_decks)
        self.counter = CardCounter(strategy='hi-lo')
        self.player_hands = []
        self.dealer_hand = None
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bets = []
        self.current_hand_index = 0
        self.game_over = False
        self.bankroll = 10000  # Start with $10,000 bankroll
        self.num_hands_played = 0
        self.results_history = []
        
    def reset(self):
        """
        Reset the game to start a new hand
        """
        # Check if deck needs shuffling
        if self.deck.needs_shuffle():
            self.deck.reset()
            self.counter.reset()
        
        self.player_hands = [Hand()]
        self.dealer_hand = Hand()
        self.bets = [self.counter.get_bet_amount(self.min_bet, self.max_bet)]
        self.current_hand_index = 0
        self.game_over = False
        
        # Deal initial cards
        for _ in range(2):
            for hand in self.player_hands:
                card = self.deck.deal()
                hand.add_card(card)
                self.counter.update_count(card)  # This now does nothing
                
            if _ == 0:  # Only deal one card to dealer initially (the visible card)
                card = self.deck.deal()
                self.dealer_hand.add_card(card)
                self.counter.update_count(card)  # This now does nothing
        
        # Update decks remaining
        self.counter.update_decks_remaining(self.deck.cards_remaining())
        
        # Get initial state
        return self._get_state()
    
    def _get_state(self):
        """
        Return current state of the game
        """
        if self.current_hand_index < len(self.player_hands):
            current_hand = self.player_hands[self.current_hand_index]
            
            # Dealer's up card value
            dealer_card = self.dealer_hand.get_first_card()
            dealer_up_card = dealer_card.get_value() if dealer_card else 0
            
            # Player's current hand value
            player_value = current_hand.value
            is_soft = current_hand.is_soft
            can_split = current_hand.can_split
            
            # Card counting information (now always 0)
            running_count = 0
            true_count = 0
            decks_remaining = self.counter.decks_remaining
            
            # Remaining card distribution
            card_distribution = self.deck.get_card_count()
            
            return {
                'dealer_up_card': dealer_up_card,
                'player_hand': current_hand,
                'player_value': player_value,
                'is_soft': is_soft,
                'can_split': can_split,
                'running_count': running_count,
                'true_count': true_count,
                'decks_remaining': decks_remaining,
                'card_distribution': card_distribution,
                'bankroll': self.bankroll,
                'bet': self.bets[self.current_hand_index]
            }
        else:
            return None
    
    def step(self, action):
        """
        Take action and return new state, reward, and done flag
        Actions: 0 = hit, 1 = stand, 2 = double, 3 = split
        """
        if self.game_over:
            return self._get_state(), 0, True
        
        if self.current_hand_index >= len(self.player_hands):
            return self._get_state(), 0, True
        
        current_hand = self.player_hands[self.current_hand_index]
        current_bet = self.bets[self.current_hand_index]
        reward = 0
        done = False
        
        # Process player action
        if action == 0:  # Hit
            card = self.deck.deal()
            current_hand.add_card(card)
            self.counter.update_count(card)  # This now does nothing
            
            if current_hand.is_busted():
                reward = -current_bet
                self.bankroll -= current_bet
                self.current_hand_index += 1
                
                if self.current_hand_index >= len(self.player_hands):
                    done = True
                    self.game_over = True
                    self._record_result("Bust", reward)
        
        elif action == 1:  # Stand
            self.current_hand_index += 1
            
            if self.current_hand_index >= len(self.player_hands):
                # All player hands are done, deal dealer cards
                done = True
                self._deal_dealer()
                reward = self._calculate_rewards()
                self.bankroll += reward
                self.game_over = True
        
        elif action == 2:  # Double
            if len(current_hand.cards) == 2:
                # Double the bet
                self.bets[self.current_hand_index] *= 2
                current_bet *= 2
                
                # Deal one more card and stand
                card = self.deck.deal()
                current_hand.add_card(card)
                self.counter.update_count(card)  # This now does nothing
                
                self.current_hand_index += 1
                
                if current_hand.is_busted():
                    reward = -current_bet
                    self.bankroll -= current_bet
                    self._record_result("Bust after Double", reward)
                
                if self.current_hand_index >= len(self.player_hands):
                    done = True
                    if not current_hand.is_busted():
                        self._deal_dealer()
                        reward = self._calculate_rewards()
                        self.bankroll += reward
                    self.game_over = True
            else:
                # Can't double after hitting, treat as hit
                card = self.deck.deal()
                current_hand.add_card(card)
                self.counter.update_count(card)  # This now does nothing
                
                if current_hand.is_busted():
                    reward = -current_bet
                    self.bankroll -= current_bet
                    self.current_hand_index += 1
                    self._record_result("Bust", reward)
                    
                    if self.current_hand_index >= len(self.player_hands):
                        done = True
                        self.game_over = True
        
        elif action == 3:  # Split
            if current_hand.can_split:
                # Create a new hand
                new_hand = Hand()
                new_hand.is_splitted = True
                
                # Move second card to new hand
                card = current_hand.cards.pop()
                new_hand.add_card(card)
                
                # Recalculate values
                current_hand._calculate_value()
                current_hand._check_split()
                
                # Deal new cards to both hands
                card = self.deck.deal()
                current_hand.add_card(card)
                self.counter.update_count(card)  # This now does nothing
                
                card = self.deck.deal()
                new_hand.add_card(card)
                self.counter.update_count(card)  # This now does nothing
                
                # Add new hand and bet
                self.player_hands.append(new_hand)
                self.bets.append(current_bet)
            else:
                # Can't split, treat as hit
                card = self.deck.deal()
                current_hand.add_card(card)
                self.counter.update_count(card)  # This now does nothing
                
                if current_hand.is_busted():
                    reward = -current_bet
                    self.bankroll -= current_bet
                    self.current_hand_index += 1
                    self._record_result("Bust", reward)
                    
                    if self.current_hand_index >= len(self.player_hands):
                        done = True
                        self.game_over = True
        
        # Update decks remaining
        self.counter.update_decks_remaining(self.deck.cards_remaining())
        
        if done:
            self.num_hands_played += 1
        
        return self._get_state(), reward, done
    
    def _deal_dealer(self):
        """
        Deal cards to dealer according to rules (hit on soft 17)
        """
        # Deal the dealer's second card
        card = self.deck.deal()
        self.dealer_hand.add_card(card)
        self.counter.update_count(card)  # This now does nothing
        
        # Hit until 17 or higher
        while self.dealer_hand.value < 17 or (self.dealer_hand.value == 17 and self.dealer_hand.is_soft):
            card = self.deck.deal()
            self.dealer_hand.add_card(card)
            self.counter.update_count(card)  # This now does nothing
    
    def _calculate_rewards(self):
        """
        Calculate rewards for all player hands
        """
        total_reward = 0
        
        for i, hand in enumerate(self.player_hands):
            bet = self.bets[i]
            
            # Save the current_hand_index temporarily
            original_index = self.current_hand_index
            # Set to the current hand being processed
            self.current_hand_index = i
            
            if hand.is_busted():
                reward = -bet
                result = "Bust"
            elif hand.is_blackjack and not self.dealer_hand.is_blackjack:
                reward = bet * 1.5  # 3:2 payout for blackjack
                result = "Blackjack"
            elif self.dealer_hand.is_busted():
                reward = bet
                result = "Dealer Bust"
            elif hand.is_blackjack and self.dealer_hand.is_blackjack:
                reward = 0  # Push
                result = "Push (Both Blackjack)"
            elif hand.value > self.dealer_hand.value:
                reward = bet
                result = "Win"
            elif hand.value < self.dealer_hand.value:
                reward = -bet
                result = "Lose"
            else:
                reward = 0  # Push
                result = "Push"
            
            total_reward += reward
            self._record_result(result, reward)
            
            # Restore the original index
            self.current_hand_index = original_index
        
        return total_reward
    
    def _record_result(self, result_type, reward):
        """
        Record game result for analysis
        """
        # Check if current_hand_index is valid before using it
        if self.current_hand_index < len(self.player_hands):
            player_hand = self.player_hands[self.current_hand_index]
        else:
            # Use the last hand if index is out of range
            player_hand = self.player_hands[-1] if self.player_hands else None
        
        self.results_history.append({
            'hand_num': self.num_hands_played,
            'result': result_type,
            'reward': reward,
            'running_count': 0,  # Always 0 now
            'true_count': 0,     # Always 0 now
            'bankroll': self.bankroll,
            'dealer_hand': str(self.dealer_hand),
            'player_hand': str(player_hand) if player_hand else "None"
        })


class BasicStrategyAgent:
    """
    Agent that uses basic strategy WITHOUT card counting
    """
    def __init__(self):
        # Basic strategy charts (simplified)
        # Format: player_value -> dealer_upcard -> action
        # Actions: 0 = hit, 1 = stand, 2 = double, 3 = split
        
        # Hard hands
        self.hard_strategy = {
            21: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            20: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            19: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            18: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            17: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            16: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            15: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            14: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            13: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            12: {2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            11: {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2},
            10: {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 0, 11: 0},
            9: {2: 0, 3: 2, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            8: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            7: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            6: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            5: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            4: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
        }
        
        # Soft hands (hands with an Ace)
        self.soft_strategy = {
            21: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            20: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            19: {2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            18: {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1, 9: 0, 10: 0, 11: 0},
            17: {2: 0, 3: 2, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            16: {2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            15: {2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            14: {2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            13: {2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
        }
        
        # Pair splitting
        self.pair_strategy = {
            'A': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3},
            'K': {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            'Q': {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            'J': {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            '10': {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1},
            '9': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 1, 8: 3, 9: 3, 10: 1, 11: 1},
            '8': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3},
            '7': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 0, 9: 0, 10: 0, 11: 0},
            '6': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            '5': {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 0, 11: 0},
            '4': {2: 0, 3: 0, 4: 0, 5: 3, 6: 3, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
            '3': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 0, 9: 0, 10: 0, 11: 0},
            '2': {2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 0, 9: 0, 10: 0, 11: 0},
        }
        
        # For tracking (no longer used but kept for compatibility)
        self.deviation_history = []
    
    def get_action(self, state):
        """
        MODIFIED: Get action using ONLY basic strategy (no count deviations)
        """
        if not state:
            return 1  # Stand if no state (should not happen)
        
        player_hand = state['player_hand']
        player_value = state['player_value']
        dealer_up_card = state['dealer_up_card']
        is_soft = state['is_soft']
        can_split = state['can_split']
        
        # Check if we can split (ignoring count completely)
        if can_split and len(player_hand.cards) == 2:
            card_value = player_hand.cards[0].value
            action = self.pair_strategy.get(card_value, {}).get(dealer_up_card, 0)
        
        # Check soft hands (ignoring count completely)
        elif is_soft:
            action = self.soft_strategy.get(player_value, {}).get(dealer_up_card, 0)
        
        # Hard hands (ignoring count completely)
        else:
            action = self.hard_strategy.get(player_value, {}).get(dealer_up_card, 0)
        
        return action