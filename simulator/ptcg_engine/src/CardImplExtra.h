// Cards added on top of the competition card pool.
//
// Kept out of CardImpl.h on purpose: that file is part of the competition-licensed
// ptcg_engine package and stays byte-identical to what shipped, the same reason
// build_engine.sh passes -include climits as a flag instead of patching a source.
// Everything local lives here and is called from All.h, before InitializeCard(),
// which needs every Stage 1 present to build its evolution map.
//
// Id allocation: card ids start at 1500, not at 1268. Core.h reserves 1268
// (NITRO_FIRE_ENERGY) and 1429 (ANGE_FLOETTE) for cards absent from this
// snapshot's table but still branched on in SelectProc.h and GameProc.h, so a
// card created at either id silently inherits that behaviour. Skill ids start at
// 500 (pool max 434), attack ids at 2000 (pool max 1556).

#pragma once

#include "CreateCard.h"

inline void CardImplExtra() {
	using enum CardType;
	using enum EnergyType;
	using enum PokemonType;
	using enum EvolutionType;
	using enum EffectType;
	using enum TargetPlayer;
	using enum ComparatorType;

	CreateCard(1500, u8"タテトプス", Pokemon, 1500)
		.nameEn(u8"Shieldon")
		.pokemon(Normal, Basic, Metal, 70, 2)
		.weakness(Fire)
		.resistance(Grass)
		.attack(2000, u8"たいあたり", u8"", "20", { Colorless })
		.textEn(u8"Tackle", u8"");

	CreateCard(1501, u8"トリデプス", Pokemon, 1501)
		.nameEn(u8"Bastiodon")
		.pokemon(Normal, Stage1, Metal, 140, 4)
		.evolvesFrom(u8"タテトプス")
		.weakness(Fire)
		.resistance(Grass)
		.abilityBattleField(500, u8"ベンチシールド", u8"このポケモンがいるかぎり、自分のベンチポケモン全員は、相手のワザのダメージを受けない。")
		.textEn(u8"Bench Shield", u8"Prevent all damage done to your Benched Pokémon by attacks from your opponent’s Pokémon.")
		.effect(NoDamageEnemyAttack, Me).targetBench()
		.attack(2001, u8"アイアンプレス", u8"", "80", { Metal, Colorless, Colorless })
		.textEn(u8"Iron Press", u8"");

	CreateCard(1502, u8"ポケモンセンターのお姉さん", Supporter, 1502)
		.nameEn(u8"Pokémon Center Lady")
		.playSkill(501, u8"自分のポケモン1匹のHPを「60」回復し、そのポケモンの特殊状態を、すべて回復する。")
		.textEn(u8"Pokémon Center Lady", u8"Heal 60 damage from 1 of your Pokémon and remove all Special Conditions from that Pokémon.")
		.effect(Heal, Me).eVal(60).targetPokemon().singleSelect()
		.separator()
		.effectEffectedCard(RecoverSpecialCondition);

	// Mega Excadrill ex (Pitch Black 65). Both attacks reuse existing patterns:
	// Undermine is Sandy Flapping's enemy-deck discard (CardImpl.h:9672), and
	// Maximum Drilling is High-Voltage Press (CardImpl.h:4595) with new numbers --
	// ConditionType::AttackEnergyExtra is the engine's purpose-built condition for
	// "N more Energy attached than this attack's cost". Drilbur is already in the
	// pool (cards 81 and 526), so no pre-evolution needed.
	CreateCard(1503, u8"メガドリュウズex", Pokemon, 1503)
		.nameEn(u8"Mega Excadrill ex")
		.pokemon(MegaEx, Stage1, Metal, 340, 4)
		.evolvesFrom(u8"モグリュー")
		.weakness(Fire)
		.resistance(Grass)
		.attack(2002, u8"アンダーマイン", u8"相手の山札を上から2枚トラッシュする。", "90", { Metal, Metal })
		.textEn(u8"Undermine", u8"Discard the top 2 cards of your opponent’s deck.")
		.postEffect(DeckToTrash, Enemy).eVal(2)
		.attack(2003, u8"マキシマムドリル", u8"このワザを使うためのエネルギーより、2個多くエネルギーがついているなら、130ダメージ追加。", "200+", { Metal, Metal, Metal })
		.textEn(u8"Maximum Drilling", u8"If this Pokémon has at least 2 extra Energy attached (in addition to this attack’s cost), this attack does 130 more damage.")
		.setPreEffect()
		.condition(ConditionType::AttackEnergyExtra, 2, GreaterEqual)
		.preEffectAttackDamageChange(130);
}
