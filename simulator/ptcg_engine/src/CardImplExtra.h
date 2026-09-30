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
//
// The 2026 block (Chaos Rising, Pitch Black) continues at card 1504, skill 503,
// attack 2004 -- except the two cards that *are* the reserved ids: Nitro Fire
// Energy is created at 1268 and Ange Floette at 1429, on purpose, so they pick
// up the behaviour SelectProc.h and GameProc.h already have for them. Each card
// reuses the DSL of an implemented card with the same wording; where one did
// not, the comment above the card names the precedent it was assembled from.
//
// One card needed a primitive the engine did not have. Bastiodon's Ancient
// Bulwark adds EffectType::NoDamageEnemyLessEqualEnergy2Attack (Types.h), its
// Card flag (Card.h), the continual case (EffectContinual.h) and the check in
// CalcDamage (SetProperty.h) -- each next to the existing attacker-conditioned
// prevention it mirrors, noDamageEnemyBasicExAttack.
//
// Left out, because the engine has no way to express them: Delibird (CRI 18),
// Tauros (CRI 69), Adversity Policy (CRI 74), Transformation Tome (CRI 83),
// Heatran (PBL 7), Manectric (PBL 24), Thievul (PBL 54), Bronzong (PBL 64),
// Backtrack Badge (PBL 74), Tremendous Bomb (PBL 82).
//
// 30th Celebration (30C, Sep 2026) follows at card 1660-1774: its 115 new
// cards (Poké Pad, Switch and Ultra Ball are reprints of pool cards). Left out
// for the same reason: Alolan Exeggutor (2), Illumise (4), Pikachu Overwriting
// Bolt (40), Seismitoad (84), Yveltal (100), Ditto (115), Snorlax (119), and
// Maushold (125), whose Tandemaus is not in the pool. Mew ex (66) was on that
// list until Memory Helix was added to the engine; it is card 1775, on the end.
//
// 1500-1502 predate the 2026 block: this Shieldon/Bastiodon pair is a Basic ->
// Stage 1 line with Bench Shield, not the Pitch Black fossil line (1506/1507).

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

	// ==== BEGIN 2026 sets: Chaos Rising (CRI), Pitch Black (PBL) ====
	// ---- Pitch Black: the Bastiodon / Rampardos ex fossil lines ----
	CreateCard(1504, u8"古びたたての化石", Item, 1504)
		.nameEn(u8"Antique Armor Fossil")
		.playSkill(503, u8"このカードは、HP60の【無】タイプの【たね】ポケモンとして、場に出せる。このカードは、特殊状態にならず、にげられない。\n自分の番の中でなら、場に出ているこのカードをトラッシュできる。")
		.textEn(u8"Antique Armor Fossil", u8"Play this card as if it were a 60-HP Basic {C} Pokémon. This card can’t be affected by any Special Conditions and can’t retreat.\n\nAt any time during your turn, you may discard this card from play.")
		.fossil()
		.abilityActive(504, u8"たてのまもり", u8"このポケモンがバトル場にいるかぎり、自分のポケモン全員が、相手のポケモンから受けるワザのダメージは「-10」される。")
		.textEn(u8" Protective Armor", u8"As long as this Pokémon is in the Active Spot, all of your Pokémon take 10 less damage from attacks from your opponent’s Pokémon (after applying Weakness and Resistance).")
		.effect(TakeEnemyAttackDamageChange, Me).eVal(-10).targetPokemon();

	CreateCard(1505, u8"古びたずがいの化石", Item, 1505)
		.nameEn(u8"Antique Skull Fossil")
		.playSkill(505, u8"このカードは、HP60の【無】タイプの【たね】ポケモンとして、場に出せる。このカードは、特殊状態にならず、にげられない。\n自分の番の中でなら、場に出ているこのカードをトラッシュできる。")
		.textEn(u8"Antique Skull Fossil", u8"Play this card as if it were a 60-HP Basic {C} Pokémon. This card can’t be affected by any Special Conditions and can’t retreat.\n\nAt any time during your turn, you may discard this card from play.")
		.fossil()
		.abilityActive(506, u8"ずがいのトゲ", u8"このポケモンが、バトル場で相手のポケモンからワザのダメージを受けたとき、ワザを使ったポケモンにダメカンを3個のせる。")
		.textEn(u8" Spiny Skull", u8"If this Pokémon is in the Active Spot and is damaged by an attack from your opponent’s Pokémon (even if this Pokémon is Knocked Out), place 3 damage counters on the Attacking Pokémon.")
		.triggerMe(TriggerType::DamagedEnemyAttackActive)
		.effectTriggerObject(DamageCounter).eVal(3);

	CreateCard(1506, u8"タテトプス", Pokemon, 1506)
		.nameEn(u8"Shieldon")
		.pokemon(Normal, Stage1, Metal, 100, 3)
		.evolvesFrom(u8"古びたたての化石")
		.weakness(Fire)
		.resistance(Grass)
		.attack(2004, u8"くだく", u8"相手のバトルポケモンについているエネルギーを1個選び、トラッシュする。", "50", { Metal, Colorless })
		.textEn(u8"Smithereen Smash", u8"Discard an Energy from your opponent’s Active Pokémon.")
		.postEffect(ToTrash, Enemy).targetAttachedEnergy().targetCondition(TargetType::AttachedActivePokemon).selectEnergy(1);

	// Ancient Bulwark needs the one new primitive in this group:
	// NoDamageEnemyLessEqualEnergy2Attack, checked in CalcDamage beside the other
	// attacker-conditioned preventions and counted with State::energyCount, the
	// same count "can't attack with 2 or less Energy" already uses.
	CreateCard(1507, u8"トリデプス", Pokemon, 1507)
		.nameEn(u8"Bastiodon")
		.pokemon(Normal, Stage2, Metal, 160, 4)
		.evolvesFrom(u8"タテトプス")
		.weakness(Fire)
		.resistance(Grass)
		.abilityBench(507, u8"たいこのぼうへき", u8"このポケモンがベンチにいるかぎり、自分のポケモン全員は、ついているエネルギーが2個以下の相手のポケモンからワザのダメージを受けない。")
		.textEn(u8" Ancient Bulwark", u8"As long as this Pokémon is on your Bench, prevent all damage done to each of your Pokémon by attacks from your opponent’s Pokémon that have 2 or less Energy attached.")
		.effect(NoDamageEnemyLessEqualEnergy2Attack, Me).targetPokemon()
		.attack(2005, u8"ぶちかます", u8"", "160", { Metal, Metal, Colorless })
		.textEn(u8"Hammer In", u8"");

	CreateCard(1508, u8"ズガイドス", Pokemon, 1508)
		.nameEn(u8"Cranidos")
		.pokemon(Normal, Stage1, Fighting, 100, 2)
		.evolvesFrom(u8"古びたずがいの化石")
		.weakness(Grass)
		.attack(2006, u8"つきとばす", u8"相手のバトルポケモンをベンチポケモンと入れ替える。［バトル場に出すポケモンは相手が選ぶ。］", "70", { Fighting, Fighting })
		.textEn(u8"Push Down", u8"Switch out your opponent’s Active Pokémon to the Bench. (Your opponent chooses the new Active Pokémon.)")
		.setPostEffect()
		.effectSwitch(Enemy).enemySelect().effectTargetActive();

	// Rowdy Hammer: DamageChangeActiveNextTurn lands in nextTurn.damageChangeActive,
	// which CalcDamage adds in step 2 (before Weakness/Resistance) only against a
	// target in the Active Spot -- exactly "to your opponent's Active Pokémon".
	CreateCard(1509, u8"ラムパルドex", Pokemon, 1509)
		.nameEn(u8"Rampardos ex")
		.pokemon(Ex, Stage2, Fighting, 330, 2)
		.evolvesFrom(u8"ズガイドス")
		.weakness(Grass)
		.activateSkillOnceTurnActive(508, u8"はかいのずつき", u8"このポケモンがバトル場にいるなら、自分の番に1回使える。コインを1回投げオモテなら、相手のバトルポケモンについているエネルギーを1個選び、トラッシュする。")
		.textEn(u8" Destructive Headbutting", u8"Once during your turn, if this Pokémon is in the Active Spot, you may use this Ability. Flip a coin. If heads, discard an Energy from your opponent’s Active Pokémon.")
		.exist(AreaType::Energy, Enemy).targetCondition(TargetType::AttachedActivePokemon)
		.effectBreakIfCoinTail()
		.effect(ToTrash, Enemy).targetAttachedEnergy().targetCondition(TargetType::AttachedActivePokemon).selectEnergy(1)
		.attack(2007, u8"ランページハンマー", u8"次の自分の番、このポケモンが使うワザの、相手のバトルポケモンへのダメージは「+150」される。", "150", { Fighting, Fighting })
		.textEn(u8"Rowdy Hammer", u8"During your next turn, attacks used by this Pokémon do 150 more damage to your opponent’s Active Pokémon (before applying Weakness and Resistance).")
		.postEffectMe(DamageChangeActiveNextTurn).eVal(150);

	CreateCard(1510, u8"化石採掘場", Stadium, 1510)
		.nameEn(u8"Fossil Quarry")
		.stadiumActivateSkillOnceTurn(509, u8"おたがいのプレイヤーは、自分の番ごとに1回、自分の山札から、名前に「古びた」とつくグッズを2枚まで選び、ベンチに出してよい。そして山札を切る。")
		.textEn(u8"Fossil Quarry", u8"Once during each player’s turn, that player may search their deck for up to 2 Item cards that have “Antique” in their name and put them onto their Bench. Then, that player shuffles their deck.")
		.notFullBench()
		.existMyDeck()
		.effect(ToBench, Me).targetDeck().targetItem().targetNameContains(u8"古びた").maxSelect(2)
		.effectShuffle();

	CreateCard(1511, u8"マグネット鋼エネルギー", SpecialEnergy, 1511)
		.nameEn(u8"Magnetic Metal Energy")
		.energySkill(510, u8"このカードは、ポケモンについているかぎり、【鋼】エネルギー1個ぶんとしてはたらく。\n\nこのカードをつけている【鋼】ポケモンは、にげるためのエネルギーが、すべてなくなる。")
		.textEn(u8"Magnetic Metal Energy", u8"As long as this card is attached to a Pokémon, it provides {M} Energy.\n\nThe {M} Pokémon this card is attached to has no Retreat Cost.")
		.effectEnergyContinual(NoRetreatCost).targetEnergyType(Metal)
		.specialEnergy(EnergyType::Metal, 1);

	// The Pokémon that went to the Bench is the switch's Effected card
	// (setTargetSwitchBench, as Scramble Switch uses it); the heal is then gated
	// on that card being a Pokémon ex, which TargetType::Ex reads as ex or Mega ex.
	CreateCard(1512, u8"AZの安らぎ", Supporter, 1512)
		.nameEn(u8"AZ’s Tranquility")
		.playSkill(511, u8"自分のバトルポケモンをベンチポケモンと入れ替える。「ポケモンex」をベンチに入れ替えた場合、そのポケモンのHPを「80」回復する。")
		.textEn(u8"AZ’s Tranquility", u8"Switch your Active Pokémon with 1 of your Benched Pokémon. If you moved a Pokémon ex to your Bench in this way, heal 80 damage from that Pokémon.")
		.effectSwitch(Me).setTargetSwitchBench()
		.separator()
		.effectEffectedCard(Heal).eVal(80).targetCondition(TargetType::Ex);

	CreateCard(1513, u8"スペシャルレッドカード", Item, 1513)
		.nameEn(u8"Special Red Card")
		.playSkill(512, u8"このカードは、相手のサイドの残り枚数が3枚以下のときにしか使えない。\n相手は相手自身の手札をすべてウラにして切り、山札の下にもどす。その後、相手は山札を3枚引く。")
		.textEn(u8"Special Red Card", u8"You can use this card only if your opponent has 3 or fewer Prize cards remaining.\n\nYour opponent shuffles their hand and puts it on the bottom of their deck. If they put any cards on the bottom of their deck in this way, they draw 3 cards.")
		.conditionLessEqual(3, Enemy).targetPrize()
		.exist(AreaType::Hand, Enemy)
		.effect(ToDeckBottomClose, Enemy).targetHand()
		.separator()
		.effect(Draw, Enemy).eVal(3);

	// ---- Mega Excadrill ex support (Pitch Black Drilbur, Chaos Rising Metagross line) ----
	CreateCard(1514, u8"モグリュー", Pokemon, 1514)
		.nameEn(u8"Drilbur")
		.pokemon(Normal, Basic, Fighting, 70, 2)
		.weakness(Grass)
		.attack(2008, u8"なかまをよぶ", u8"自分の山札から【たね】ポケモンを2枚まで選び、ベンチに出す。そして山札を切る。", "", { Colorless })
		.textEn(u8"Call for Family", u8"Search your deck for up to 2 Basic Pokémon and put them onto your Bench. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToBenchAndShuffle(2).targetBasicPokemon()
		.attack(2009, u8"ツメをたてる", u8"", "50", { Colorless, Colorless, Colorless })
		.textEn(u8"Dig Claws", u8"");

	CreateCard(1515, u8"ダンバル", Pokemon, 1515)
		.nameEn(u8"Beldum")
		.pokemon(Normal, Basic, Metal, 70, 1)
		.weakness(Fire)
		.resistance(Grass)
		.attack(2010, u8"ずつき", u8"", "10", { Metal })
		.textEn(u8"Headbutt", u8"")
		.attack(2011, u8"ビーム", u8"", "20", { Metal, Colorless })
		.textEn(u8"Beam", u8"");

	CreateCard(1516, u8"メタング", Pokemon, 1516)
		.nameEn(u8"Metang")
		.pokemon(Normal, Stage1, Metal, 100, 2)
		.evolvesFrom(u8"ダンバル")
		.weakness(Fire)
		.resistance(Grass)
		.attack(2012, u8"メタルクロー", u8"", "30", { Metal })
		.textEn(u8"Metal Claw", u8"")
		.attack(2013, u8"ガードプレス", u8"次の相手の番、このポケモンが受けるワザのダメージは「-30」される。", "70", { Metal, Metal, Colorless })
		.textEn(u8"Guard Press", u8"During your opponent’s next turn, this Pokémon takes 30 less damage from attacks (after applying Weakness and Resistance).")
		.postEffectMe(TakeDamageChangeNextEnemyTurn).eVal(-30);

	CreateCard(1517, u8"メタグロス", Pokemon, 1517)
		.nameEn(u8"Metagross")
		.pokemon(Normal, Stage2, Metal, 180, 3)
		.evolvesFrom(u8"メタング")
		.weakness(Fire)
		.resistance(Grass)
		.attack(2014, u8"はねかえす", u8"相手のバトルポケモンをベンチポケモンと入れ替える。［バトル場に出すポケモンは相手が選ぶ。］", "60", { Metal })
		.textEn(u8"Bounce Back", u8"Switch out your opponent’s Active Pokémon to the Bench. (Your opponent chooses the new Active Pokémon.)")
		.setPostEffect()
		.effectSwitch(Enemy).enemySelect().effectTargetActive()
		.attack(2015, u8"メタリックハンマー", u8"のぞむなら、このポケモンについている【鋼】エネルギーを3個トラッシュし、150ダメージ追加。", "150+", { Metal, Metal, Metal, Colorless })
		.textEn(u8"Metallic Hammer", u8"You may discard 3 {M} Energy from this Pokémon and have this attack do 150 more damage.")
		.preEffectSelectActivate()
		.effect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Metal).selectEnergy(3)
		.preEffectAttackDamageChange(150);

	// ---- Chaos Rising: the Mega Greninja ex line ----
	CreateCard(1518, u8"ケロマツ", Pokemon, 1518)
		.nameEn(u8"Froakie")
		.pokemon(Normal, Basic, Water, 70, 1)
		.weakness(Lightning)
		.attack(2016, u8"もってくる", u8"自分の山札を1枚引く。", "", { Colorless })
		.textEn(u8"Collect", u8"Draw a card.")
		.setPostEffect()
		.effectDraw(1)
		.attack(2017, u8"みずでっぽう", u8"", "10", { Water })
		.textEn(u8"Water Gun", u8"");

	CreateCard(1519, u8"ゲコガシラ", Pokemon, 1519)
		.nameEn(u8"Frogadier")
		.pokemon(Normal, Stage1, Water, 100, 1)
		.evolvesFrom(u8"ケロマツ")
		.weakness(Lightning)
		.attack(2018, u8"よびよせのじゅつ", u8"自分の山札からポケモンを3枚まで選び、相手に見せて、手札に加える。そして山札を切る。", "", { Water })
		.textEn(u8"Summoning Jutsu", u8"Search your deck for up to 3 Pokémon, reveal them, and put them into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToHandAndShuffle(3).targetPokemonCard()
		.attack(2019, u8"アクアエッジ", u8"", "50", { Water, Water })
		.textEn(u8"Aqua Edge", u8"");

	// Mortal Shuriken is Volcarona's hand-cost ability (costHandTrashLimited on
	// a Basic {W} Energy) on an Active-only once-per-turn skill, then Bramblin's
	// "damage counters on 1 of your opponent's Pokémon" with 6.
	CreateCard(1520, u8"メガゲッコウガex", Pokemon, 1520)
		.nameEn(u8"Mega Greninja ex")
		.pokemon(MegaEx, Stage2, Water, 350, 1)
		.evolvesFrom(u8"ゲコガシラ")
		.weakness(Lightning)
		.activateSkillOnceTurnActive(513, u8"ひっさつしゅりけん", u8"このポケモンがバトル場にいて、自分の番に、自分の手札から「基本【水】エネルギー」を1枚トラッシュするなら、1回使える。相手のポケモン1匹に、ダメカンを6個のせる。")
		.textEn(u8" Mortal Shuriken", u8"Once during your turn, if this Pokémon is in the Active Spot, you may discard a Basic {W} Energy card from your hand in order to use this Ability. Place 6 damage counters on 1 of your opponent’s Pokémon.")
		.costHandTrashLimited(1).targetCardId(WATER_ENERGY)
		.effect(DamageCounter, Enemy).eVal(6).targetPokemon().singleSelect()
		.attack(2020, u8"ニンジャスピナー", u8"のぞむなら、このポケモンについている【水】エネルギーを1個手札にもどし、80ダメージ追加。", "120+", { Water, Water })
		.textEn(u8"Ninja Spinner", u8"You may put a {W} Energy attached to this Pokémon into your hand and have this attack do 80 more damage.")
		.preEffectSelectActivate()
		.effect(ToHand, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Water).selectEnergy(1)
		.preEffectAttackDamageChange(80);

	// Ange Floette (CRI 75) is only playable onto this stadium; GameProc.h
	// matches it by the Japanese name, so the name here must stay プリズムタワー.
	CreateCard(1521, u8"プリズムタワー", Stadium, 1521)
		.nameEn(u8"Prism Tower")
		.stadiumActivateSkillOnceTurn(514, u8"おたがいのプレイヤーは、自分の番ごとに1回、自分の手札を2枚トラッシュするなら、自分の山札を1枚引いてよい。")
		.textEn(u8"Prism Tower", u8"Once during each player’s turn, that player may discard 2 cards from their hand in order to draw a card.")
		.existMyDeck()
		.costHandTrashLimited(2)
		.effect(Draw, Me).eVal(1);

	// ---- Pitch Black: Hide 'n' Sneak (Dhelmise) ----
	// Hide 'n' Sneak is both halves of the existing single-sided preventions:
	// Skeledirge's NoEffectEnemyAttack and Mega Clefable ex's NoEnemyAbility (at
	// its priority, so it is in place before an opposing Ability resolves).
	// The Hide 'n' Sneak counters below match it by its Japanese name, ばけがくれ.
	CreateCard(1522, u8"チャデス", Pokemon, 1522)
		.nameEn(u8"Poltchageist")
		.pokemon(Normal, Basic, Grass, 30, 0)
		.weakness(Fire)
		.abilityBattleField(515, u8"ばけがくれ", u8"このポケモンは、相手のワザや特性の効果を受けない。")
		.textEn(u8" Hide ’n’ Sneak", u8"Prevent all effects of your opponent’s Pokémon’s attacks and Abilities done to this Pokémon. (Damage is not an effect.)")
		.effectMe(NoEffectEnemyAttack)
		.effectMe(NoEnemyAbility).priority(3)
		.attack(2021, u8"ひっそりのせる", u8"相手のバトルポケモンに、ダメカンを1個のせる。", "", { Colorless })
		.textEn(u8"Furtive Drop", u8"Place 1 damage counter on your opponent’s Active Pokémon.")
		.postEffectActiveEnemy(DamageCounter).eVal(1);

	CreateCard(1523, u8"ヤバソチャ", Pokemon, 1523)
		.nameEn(u8"Sinistcha")
		.pokemon(Normal, Stage1, Grass, 60, 1)
		.evolvesFrom(u8"チャデス")
		.weakness(Fire)
		.abilityBattleField(516, u8"ばけがくれ", u8"このポケモンは、相手のワザや特性の効果を受けない。")
		.textEn(u8" Hide ’n’ Sneak", u8"Prevent all effects of your opponent’s Pokémon’s attacks and Abilities done to this Pokémon. (Damage is not an effect.)")
		.effectMe(NoEffectEnemyAttack)
		.effectMe(NoEnemyAbility).priority(3)
		.attack(2022, u8"まっちゃスピン", u8"自分のトラッシュに、特性「ばけがくれ」を持つポケモンが6枚以上あるなら、相手のポケモン全員に、それぞれダメカンを4個のせる。", "", { Colorless })
		.textEn(u8"Matcha Spin", u8"If you have 6 or more Pokémon that have the Hide ’n’ Sneak Ability in your discard pile, place 4 damage counters on each of your opponent’s Pokémon.")
		.setPostEffect()
		.conditionGreaterEqual(6).targetTrash().targetNameCondition(TargetType::HasAbilityName, u8"ばけがくれ")
		.effect(DamageCounter, Enemy).eVal(4).targetPokemon();

	CreateCard(1524, u8"カゲボウズ", Pokemon, 1524)
		.nameEn(u8"Shuppet")
		.pokemon(Normal, Basic, Psychic, 50, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.abilityBattleField(517, u8"ばけがくれ", u8"このポケモンは、相手のワザや特性の効果を受けない。")
		.textEn(u8" Hide ’n’ Sneak", u8"Prevent all effects of your opponent’s Pokémon’s attacks and Abilities done to this Pokémon. (Damage is not an effect.)")
		.effectMe(NoEffectEnemyAttack)
		.effectMe(NoEnemyAbility).priority(3)
		.attack(2023, u8"ぶらさがる", u8"", "10", { Psychic })
		.textEn(u8"Hang Down", u8"");

	CreateCard(1525, u8"ジュペッタ", Pokemon, 1525)
		.nameEn(u8"Banette")
		.pokemon(Normal, Stage1, Psychic, 80, 1)
		.evolvesFrom(u8"カゲボウズ")
		.weakness(Darkness)
		.resistance(Fighting)
		.abilityBattleField(518, u8"ばけがくれ", u8"このポケモンは、相手のワザや特性の効果を受けない。")
		.textEn(u8" Hide ’n’ Sneak", u8"Prevent all effects of your opponent’s Pokémon’s attacks and Abilities done to this Pokémon. (Damage is not an effect.)")
		.effectMe(NoEffectEnemyAttack)
		.effectMe(NoEnemyAbility).priority(3)
		.attack(2024, u8"にんぎょうキャッチ", u8"のぞむなら、自分の山札から好きなカードを1枚選び、手札に加える。そして山札を切る。", "80", { Psychic })
		.textEn(u8"Puppet Pull", u8"You may search your deck for a card and put it into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.existMyDeck()
		.postEffectSelectActivate()
		.effectDeckToHandReverseAndShuffle(1);

	CreateCard(1526, u8"ダダリン", Pokemon, 1526)
		.nameEn(u8"Dhelmise")
		.pokemon(Normal, Basic, Psychic, 140, 3)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2025, u8"むねんのイカリ", u8"自分のトラッシュに、特性「ばけがくれ」を持つポケモンが4枚以上あるなら、140ダメージ追加。", "30+", { Psychic })
		.textEn(u8"Vengeful Anchor", u8"If you have 4 or more Pokémon that have the Hide ’n’ Sneak Ability in your discard pile, this attack does 140 more damage.")
		.setPreEffect()
		.conditionGreaterEqual(4).targetTrash().targetNameCondition(TargetType::HasAbilityName, u8"ばけがくれ")
		.preEffectAttackDamageChange(140);

	// "Quadruple" is N's Vanilluxe's DamageCounterDouble applied twice to the same
	// two Pokémon: the second pass reads the already-doubled count.
	CreateCard(1527, u8"ミカルゲ", Pokemon, 1527)
		.nameEn(u8"Spiritomb")
		.pokemon(Normal, Basic, Psychic, 60, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2026, u8"たましいエンド", u8"自分のトラッシュに、特性「ばけがくれ」を持つポケモンが13枚以上あるなら、相手のポケモンを2匹選び、それぞれのっているダメカンの数が4倍になるように、ダメカンをのせる。", "", { Psychic })
		.textEn(u8"Spiritual End", u8"If you have 13 or more Pokémon that have the Hide ’n’ Sneak Ability in your discard pile, choose 2 of your opponent’s Pokémon and quadruple the number of damage counters on each of them.")
		.setPostEffect()
		.conditionGreaterEqual(13).targetTrash().targetNameCondition(TargetType::HasAbilityName, u8"ばけがくれ")
		.effect(DamageCounterDouble, Enemy).eVal(10).targetPokemon().multiSelect(2)
		.effectEffectedCard(DamageCounterDouble).eVal(10);

	// Draw 3 per discarded card: DrawTargetCount draws one per target, so it is
	// issued three times over the same (notUpdateTarget) discard list, as
	// Meddling Memo issues it once.
	CreateCard(1528, u8"ムク", Supporter, 1528)
		.nameEn(u8"Gwynn")
		.playSkill(519, u8"自分の手札からポケモン（「ルールを持つポケモン」をのぞく）を2枚までトラッシュし、その枚数×3枚ぶん、山札を引く。")
		.textEn(u8"Gwynn", u8"Discard up to 2 Pokémon that don’t have a Rule Box from your hand, and draw 3 cards for each card you discarded in this way. (Pokémon ex, Pokémon V, etc. have Rule Boxes.)")
		.existMyHand().notMe().targetPokemonCard().targetNotRulePokemon()
		.effect(ToTrash, Me).targetHand().notMe().targetPokemonCard().targetNotRulePokemon().maxSelect(2)
		.separator()
		.effect(DrawTargetCount, Me).notUpdateTarget()
		.effect(DrawTargetCount, Me).notUpdateTarget()
		.effect(DrawTargetCount, Me).notUpdateTarget();

	// ---- Pitch Black: the Mega Chandelure ex line ----
	CreateCard(1529, u8"ヒトモシ", Pokemon, 1529)
		.nameEn(u8"Litwick")
		.pokemon(Normal, Basic, Psychic, 70, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2027, u8"おにび", u8"", "20", { Psychic })
		.textEn(u8"Will-O-Wisp", u8"");

	CreateCard(1530, u8"ランプラー", Pokemon, 1530)
		.nameEn(u8"Lampent")
		.pokemon(Normal, Stage1, Psychic, 90, 1)
		.evolvesFrom(u8"ヒトモシ")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2028, u8"ふえるあかり", u8"自分の山札から「ランプラー」を3枚まで選び、ベンチに出す。そして山札を切る。", "", { Psychic })
		.textEn(u8"Spreading Light", u8"Search your deck for up to 3 Lampent and put them onto your Bench. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToBenchAndShuffle(3).targetName(u8"ランプラー");

	CreateCard(1531, u8"メガシャンデラex", Pokemon, 1531)
		.nameEn(u8"Mega Chandelure ex")
		.pokemon(MegaEx, Stage2, Psychic, 350, 2)
		.evolvesFrom(u8"ランプラー")
		.weakness(Darkness)
		.resistance(Fighting)
		.abilityBattleField(520, u8"じゅばくのほのお", u8"このポケモンがいるかぎり、相手のバトルポケモンは、にげるためのエネルギーが1個ぶん多くなる。")
		.textEn(u8" Binding Flame", u8"Your opponent’s Active Pokémon’s Retreat Cost is {C} more.")
		.effect(RetreatCostChange, Enemy).eVal(1).targetActive()
		.attack(2029, u8"ファントムメイズ", u8"相手のバトルポケモンのにげるためのエネルギーの数×50ダメージ追加。", "130+", { Psychic, Psychic })
		.textEn(u8"Phantom Maze", u8"This attack does 50 more damage for each {C} in your opponent’s Active Pokémon’s Retreat Cost.")
		.preEffect(AttackDamageChangeRetreatCost, Enemy).eVal(50).targetActive();

	// ---- Chaos Rising / Pitch Black: the rest of both sets ----
	CreateCard(1532, u8"ビードル", Pokemon, 1532)
		.nameEn(u8"Weedle")
		.pokemon(Normal, Basic, Grass, 50, 1)
		.weakness(Fire)
		.attack(2030, u8"ふいをつく", u8"コインを1回投げウラなら、このワザは失敗。", "30", { Grass })
		.textEn(u8"Surprise Attack", u8"Flip a coin. If tails, this attack does nothing.")
		.preEffectFailAttackCoinTail();

	CreateCard(1533, u8"コクーン", Pokemon, 1533)
		.nameEn(u8"Kakuna")
		.pokemon(Normal, Stage1, Grass, 80, 3)
		.evolvesFrom(u8"ビードル")
		.weakness(Fire)
		.abilityBattleField(521, u8"かたいからだ", u8"このポケモンが受けるワザのダメージは「-20」される。")
		.textEn(u8" Exoskeleton", u8"This Pokémon takes 20 less damage from attacks (after applying Weakness and Resistance).")
		.effectMe(TakeDamageChange).eVal(-20)
		.attack(2031, u8"ぶらさがる", u8"", "20", { Grass })
		.textEn(u8"Hang Down", u8"");

	// Beedrill and Beedrill ex both contain スピアー, the same name-substring match
	// the engine uses for other "X (including X ex)" counts.
	CreateCard(1534, u8"スピアーex", Pokemon, 1534)
		.nameEn(u8"Beedrill ex")
		.pokemon(Ex, Stage2, Grass, 310, 1)
		.evolvesFrom(u8"コクーン")
		.weakness(Fire)
		.attack(2032, u8"ビーランブル", u8"自分の場の「スピアー（『ポケモンex』をふくむ）」の数×110ダメージ。", "110×", { Grass })
		.textEn(u8"Rumbling Bees", u8"This attack does 110 damage for each of your Beedrill and Beedrill ex in play.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(110).targetPokemon().targetNameContains(u8"スピアー");

	CreateCard(1535, u8"マスキッパ", Pokemon, 1535)
		.nameEn(u8"Carnivine")
		.pokemon(Normal, Basic, Grass, 110, 2)
		.weakness(Fire)
		.attack(2033, u8"まるかじり", u8"相手のバトルポケモンのにげるためのエネルギーがないなら、80ダメージ追加。", "80+", { Colorless, Colorless, Colorless })
		.textEn(u8"Chomp Whole", u8"If your opponent’s Active Pokémon has no Retreat Cost, this attack does 80 more damage.")
		.setPreEffect()
		.exist(AreaType::Active, Enemy).targetCondition(TargetType::RetreatCost, 0)
		.preEffectAttackDamageChange(80);

	CreateCard(1536, u8"ハリマロン", Pokemon, 1536)
		.nameEn(u8"Chespin")
		.pokemon(Normal, Basic, Grass, 70, 2)
		.weakness(Fire)
		.attack(2034, u8"たたく", u8"", "10", { Grass })
		.textEn(u8"Beat", u8"")
		.attack(2035, u8"トゲでさす", u8"", "30", { Grass, Grass })
		.textEn(u8"Spike Sting", u8"");

	CreateCard(1537, u8"ハリボーグ", Pokemon, 1537)
		.nameEn(u8"Quilladin")
		.pokemon(Normal, Stage1, Grass, 100, 3)
		.evolvesFrom(u8"ハリマロン")
		.weakness(Fire)
		.attack(2036, u8"リーフチャージ", u8"自分の山札から「基本【草】エネルギー」を1枚選び、このポケモンにつける。そして山札を切る。", "20", { Grass })
		.textEn(u8"Leafy Charge", u8"Search your deck for a Basic {G} Energy card and attach it to this Pokémon. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckAttachEnergyMeAndShuffle(1).targetCardId(GRASS_ENERGY)
		.attack(2037, u8"つるのムチ", u8"", "80", { Grass, Grass, Colorless })
		.textEn(u8"Vine Whip", u8"");

	CreateCard(1538, u8"ブリガロン", Pokemon, 1538)
		.nameEn(u8"Chesnaught")
		.pokemon(Normal, Stage2, Grass, 180, 4)
		.evolvesFrom(u8"ハリボーグ")
		.weakness(Fire)
		.abilityActive(522, u8"ニードルアーマー", u8"このポケモンが、バトル場で相手のポケモンからワザのダメージを受けたとき、このポケモンについている【草】エネルギーの数×3個ぶんのダメカンを、ワザを使ったポケモンにのせる。")
		.textEn(u8" Needly Armor", u8"If this Pokémon is in the Active Spot and is damaged by an attack from your opponent’s Pokémon (even if this Pokémon is Knocked Out), place 3 damage counters on the Attacking Pokémon for each {G} Energy attached to this Pokémon.")
		.triggerMe(TriggerType::DamagedEnemyAttackActive)
		.effectTriggerObject(DamageCounterTypeEnergyCountMe).eVal((int)EnergyType::Grass, 3)
		.attack(2038, u8"かこいこむ", u8"次の相手の番、このワザを受けたポケモンは、にげられない。", "160", { Grass, Grass, Colorless })
		.textEn(u8"Impound", u8"During your opponent’s next turn, the Defending Pokémon can’t retreat.")
		.postEffectActiveEnemy(CannotRetreatNextTurn);

	CreateCard(1539, u8"ロコン", Pokemon, 1539)
		.nameEn(u8"Vulpix")
		.pokemon(Normal, Basic, Fire, 70, 1)
		.weakness(Water)
		.attack(2039, u8"こがす", u8"相手のバトルポケモンを【やけど】にする。", "", { Fire })
		.textEn(u8"Singe", u8"Your opponent’s Active Pokémon is now Burned.")
		.postEffect(Burn, Enemy);

	CreateCard(1540, u8"キュウコン", Pokemon, 1540)
		.nameEn(u8"Ninetales")
		.pokemon(Normal, Stage1, Fire, 120, 1)
		.evolvesFrom(u8"ロコン")
		.weakness(Water)
		.attack(2040, u8"きゅうびうつし", u8"自分のベンチポケモンを1匹選び、選んだポケモンにのっているダメカンをすべて、相手のバトルポケモンにのせ替える。", "", { Fire })
		.textEn(u8"Nine-Tailed Transfer", u8"Move all damage counters from 1 of your Benched Pokémon to your opponent’s Active Pokémon.")
		.setPostEffect()
		.existMyBench().targetDamaged()
		.effect(RemoveDamageCounterAll, Me).targetBench().targetDamaged().singleSelect()
		.effect(DamageCounterRemoved, Enemy).targetActive()
		.attack(2041, u8"おにび", u8"", "70", { Fire, Fire })
		.textEn(u8"Will-O-Wisp", u8"");

	CreateCard(1541, u8"ホウオウ", Pokemon, 1541)
		.nameEn(u8"Ho-Oh")
		.pokemon(Normal, Basic, Fire, 130, 2)
		.weakness(Water)
		.attack(2042, u8"さいきのほのお", u8"自分のトラッシュから【たね】ポケモンを3枚まで選び、ベンチに出す。", "", { Fire })
		.textEn(u8"Flames of Revival", u8"Put up to 3 Basic Pokémon from your discard pile onto your Bench.")
		.setPostEffect()
		.notFullBench()
		.effect(ToBench, Me).targetTrash().targetBasicPokemon().maxSelect(3)
		.attack(2043, u8"ぐれんのつばさ", u8"このポケモンについている【炎】エネルギーを1個選び、トラッシュする。", "130", { Fire, Fire, Fire })
		.textEn(u8"Bright Wing", u8"Discard a {R} Energy from this Pokémon.")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Fire).selectEnergy(1);

	CreateCard(1542, u8"フォッコ", Pokemon, 1542)
		.nameEn(u8"Fennekin")
		.pokemon(Normal, Basic, Fire, 70, 1)
		.weakness(Water)
		.attack(2044, u8"なかまをよぶ", u8"自分の山札から【たね】ポケモンを2枚まで選び、ベンチに出す。そして山札を切る。", "", { Colorless })
		.textEn(u8"Call for Family", u8"Search your deck for up to 2 Basic Pokémon and put them onto your Bench. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToBenchAndShuffle(2).targetBasicPokemon()
		.attack(2045, u8"ひをはく", u8"", "10", { Fire })
		.textEn(u8"Steady Firebreathing", u8"");

	CreateCard(1543, u8"テールナー", Pokemon, 1543)
		.nameEn(u8"Braixen")
		.pokemon(Normal, Stage1, Fire, 100, 1)
		.evolvesFrom(u8"フォッコ")
		.weakness(Water)
		.attack(2046, u8"かえんほうしゃ", u8"このポケモンについているエネルギーを1個選び、トラッシュする。", "80", { Fire, Fire })
		.textEn(u8"Flamethrower", u8"Discard an Energy from this Pokémon.")
		.postEffectTrashEnergyMe(1);

	CreateCard(1544, u8"マフォクシー", Pokemon, 1544)
		.nameEn(u8"Delphox")
		.pokemon(Normal, Stage2, Fire, 160, 2)
		.evolvesFrom(u8"テールナー")
		.weakness(Water)
		.activateSkillOnceTurn(523, u8"フレアマジック", u8"自分の番に、自分の手札から「基本【炎】エネルギー」を1枚トラッシュするなら、1回使える。自分の手札が7枚になるように、山札を引く。")
		.textEn(u8" Flaring Magic", u8"Once during your turn, you may discard a Basic {R} Energy card from your hand in order to use this Ability. Draw cards until you have 7 cards in your hand.")
		.existMyDeck()
		.conditionLessEqual(7).targetHand()
		.costHandTrashLimited(1).targetCardId(FIRE_ENERGY)
		.effect(DrawUntil, Me).eVal(7)
		.attack(2047, u8"エナジーストーム", u8"おたがいのポケモン全員についているエネルギーの数×30ダメージ。", "30×", { Fire, Fire })
		.textEn(u8"Energized Storm", u8"This attack does 30 damage for each Energy attached to all Pokémon.")
		.preEffect(AttackDamageChangeEnergyCount, Both).eVal(30).targetPokemon();

	CreateCard(1545, u8"シシコ", Pokemon, 1545)
		.nameEn(u8"Litleo")
		.pokemon(Normal, Basic, Fire, 70, 1)
		.weakness(Water)
		.attack(2048, u8"たいあたり", u8"", "10", { Colorless })
		.textEn(u8"Tackle", u8"");

	CreateCard(1546, u8"メガカエンジシex", Pokemon, 1546)
		.nameEn(u8"Mega Pyroar ex")
		.pokemon(MegaEx, Stage1, Fire, 340, 2)
		.evolvesFrom(u8"シシコ")
		.weakness(Water)
		.attack(2049, u8"ほえたてる", u8"次の相手の番、このワザを受けたポケモンが使うワザのダメージは「-50」される。", "80", { Fire, Colorless })
		.textEn(u8"Ferocious Bellow", u8"During your opponent’s next turn, attacks used by the Defending Pokémon do 50 less damage (before applying Weakness and Resistance).")
		.postEffect(DamageChangeNextTurn, Enemy).eVal(-50).targetActive()
		.attack(2050, u8"ビッグバンファイヤー", u8"このポケモンにのっているダメカンの数×10ダメージぶん、このワザのダメージは小さくなる。", "290-", { Fire, Fire, Colorless })
		.textEn(u8"Fiery Big Bang", u8"This attack does 10 less damage for each damage counter on this Pokémon.")
		.preEffectMe(AttackDamageChangeDamageCounter).eVal(-10);

	CreateCard(1547, u8"テッポウオ", Pokemon, 1547)
		.nameEn(u8"Remoraid")
		.pokemon(Normal, Basic, Water, 70, 1)
		.weakness(Lightning)
		.attack(2051, u8"するどいひれ", u8"", "20", { Water })
		.textEn(u8"Sharp Fin", u8"");

	CreateCard(1548, u8"オクタン", Pokemon, 1548)
		.nameEn(u8"Octillery")
		.pokemon(Normal, Stage1, Water, 110, 2)
		.evolvesFrom(u8"テッポウオ")
		.weakness(Lightning)
		.attack(2052, u8"すみふんしゃ", u8"次の相手の番、このワザを受けたポケモンがワザを使うとき、相手はコインを2回投げる。1回でもウラなら、そのワザは失敗。", "30", { Water })
		.textEn(u8"Jet of Ink", u8"During your opponent’s next turn, if the Defending Pokémon tries to use an attack, your opponent flips 2 coins. If either of them is tails, that attack doesn’t happen.")
		.postEffectActiveEnemy(AttackCoin2NextTurn)
		.attack(2053, u8"あばれまわる", u8"このポケモンを【こんらん】にする。", "120", { Water, Colorless })
		.textEn(u8"Tantrum", u8"This Pokémon is now Confused.")
		.postEffectMe(Confuse);

	CreateCard(1549, u8"ケルディオ", Pokemon, 1549)
		.nameEn(u8"Keldeo")
		.pokemon(Normal, Basic, Water, 110, 1)
		.weakness(Lightning)
		.attack(2054, u8"つきぬける", u8"相手のベンチポケモン1匹にも、20ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "20", { Water })
		.textEn(u8"Shoot Through", u8"This attack also does 20 damage to 1 of your opponent’s Benched Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffectDamageBench(20)
		.attack(2055, u8"エネリフレクト", u8"このポケモンについているエネルギーを1個選び、ベンチポケモンにつけ替える。", "70", { Water, Water })
		.textEn(u8"Reflect Energy", u8"Move an Energy from this Pokémon to 1 of your Benched Pokémon.")
		.setPostEffect()
		.existMyBench()
		.effect(SelectSwitchEnergy, Me).targetEnergyCardMe().selectEnergy(1)
		.effect(EnergySwitchEach, Me).eachSelectedList().targetBench().singleSelect();

	CreateCard(1550, u8"カチコール", Pokemon, 1550)
		.nameEn(u8"Bergmite")
		.pokemon(Normal, Basic, Water, 80, 2)
		.weakness(Metal)
		.attack(2056, u8"ひんやり", u8"", "10", { Water })
		.textEn(u8"Chilly", u8"")
		.attack(2057, u8"こおりのいぶき", u8"", "50", { Water, Colorless, Colorless })
		.textEn(u8"Frost Breath", u8"");

	CreateCard(1551, u8"クレベース", Pokemon, 1551)
		.nameEn(u8"Avalugg")
		.pokemon(Normal, Stage1, Water, 160, 4)
		.evolvesFrom(u8"カチコール")
		.weakness(Metal)
		.attack(2058, u8"ひょうざんくずし", u8"自分の山札を上から6枚トラッシュし、その中にある「基本【水】エネルギー」の枚数×60ダメージ。", "60×", { Water })
		.textEn(u8"Iceberg Breaker", u8"Discard the top 6 cards of your deck, and this attack does 60 damage for each Basic {W} Energy card you discarded in this way.")
		.setPostEffect()
		.existMyDeck()
		.preEffect(DeckToTrash, Me).eVal(6)
		.effectEffectedCard(AttackDamageChangeTargetCount).eVal(60).targetCardId(WATER_ENERGY)
		.attack(2059, u8"フロストスタンプ", u8"", "160", { Water, Water, Colorless, Colorless })
		.textEn(u8"Frost Stamp", u8"");

	CreateCard(1552, u8"コソクムシ", Pokemon, 1552)
		.nameEn(u8"Wimpod")
		.pokemon(Normal, Basic, Water, 70, 2)
		.weakness(Lightning)
		.attack(2060, u8"かじる", u8"", "10", { Water })
		.textEn(u8"Gnaw", u8"")
		.attack(2061, u8"どつく", u8"", "20", { Colorless, Colorless })
		.textEn(u8"Corkscrew Punch", u8"");

	CreateCard(1553, u8"グソクムシャ", Pokemon, 1553)
		.nameEn(u8"Golisopod")
		.pokemon(Normal, Stage1, Water, 140, 2)
		.evolvesFrom(u8"コソクムシ")
		.weakness(Lightning)
		.attack(2062, u8"きゅうしょぎり", u8"このワザのダメージで、相手のポケモンがきぜつしたなら、次の相手の番、このポケモンはワザのダメージや効果を受けない。", "30", { Water })
		.textEn(u8"Critical Slash", u8"If your opponent’s Pokémon is Knocked Out by damage from this attack, during your opponent’s next turn, prevent all damage from and effects of attacks done to this Pokémon.")
		.koNoDamageAndEffectAttackNextEnemyTurn()
		.attack(2063, u8"そこぢから", u8"次の自分の番、このポケモンはワザが使えない。", "150", { Colorless, Colorless, Colorless })
		.textEn(u8"Boundless Power", u8"During your next turn, this Pokémon can’t use attacks.")
		.postEffectMe(CannotAttackNextTurn);

	CreateCard(1554, u8"メリープ", Pokemon, 1554)
		.nameEn(u8"Mareep")
		.pokemon(Normal, Basic, Lightning, 70, 2)
		.weakness(Fighting)
		.attack(2064, u8"でんじは", u8"コインを1回投げオモテなら、相手のバトルポケモンを【マヒ】にする。", "20", { Lightning, Colorless })
		.textEn(u8"Thunder Wave", u8"Flip a coin. If heads, your opponent’s Active Pokémon is now Paralyzed.")
		.postEffectParalyzeIfCoinHead();

	CreateCard(1555, u8"モココ", Pokemon, 1555)
		.nameEn(u8"Flaaffy")
		.pokemon(Normal, Stage1, Lightning, 90, 2)
		.evolvesFrom(u8"メリープ")
		.weakness(Fighting)
		.attack(2065, u8"でんじしょうがい", u8"次の相手の番、相手は手札からグッズを出して使えない。", "40", { Lightning, Colorless })
		.textEn(u8"Disconnect", u8"During your opponent’s next turn, they can’t play any Item cards from their hand.")
		.postEffect(CannotPlayItemNextTurn, Enemy);

	// Synchro Pulse compares hand sizes with CompareCountTargetMeEnemy, the
	// condition the pool's "more Prize cards than your opponent" checks use.
	CreateCard(1556, u8"デンリュウ", Pokemon, 1556)
		.nameEn(u8"Ampharos")
		.pokemon(Normal, Stage2, Lightning, 160, 2)
		.evolvesFrom(u8"モココ")
		.weakness(Fighting)
		.abilityBattleField(524, u8"シンクロパルス", u8"自分の手札と相手の手札が同じ枚数なら、このポケモンが使うワザの、相手のバトルポケモンへのダメージは「+80」される。")
		.textEn(u8" Synchro Pulse", u8"If you have the same number of cards in your hand as your opponent, attacks used by this Pokémon do 80 more damage to your opponent’s Active Pokémon (before applying Weakness and Resistance).")
		.condition(ConditionType::CompareCountTargetMeEnemy, 0, ComparatorType::Equal).targetPlayer(TargetPlayer::Both).targetHand()
		.effectMe(DamageChangeActive).eVal(80)
		.attack(2066, u8"フラッシュボルト", u8"次の自分の番、このポケモンは「フラッシュボルト」が使えない。", "140", { Lightning, Colorless })
		.textEn(u8"Flashing Bolt", u8"During your next turn, this Pokémon can’t use Flashing Bolt.")
		.postEffectMe(CannotUseThisAttackNextTurn);

	CreateCard(1557, u8"エモンガ", Pokemon, 1557)
		.nameEn(u8"Emolga")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2067, u8"ちいさなおつかい", u8"自分の山札から基本エネルギーを2枚まで選び、相手に見せて、手札に加える。そして山札を切る。", "", { Colorless })
		.textEn(u8"Minor Errand-Running", u8"Search your deck for up to 2 Basic Energy cards, reveal them, and put them into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToHandAndShuffle(2).targetBasicEnergy()
		.attack(2068, u8"スカイリターン", u8"このポケモンと、ついているすべてのカードを、手札にもどす。", "30", { Lightning })
		.textEn(u8"Sky Return", u8"Put this Pokémon and all attached cards into your hand.")
		.postEffectMe(ToHandWithAttach);

	CreateCard(1558, u8"デオキシス", Pokemon, 1558)
		.nameEn(u8"Deoxys")
		.pokemon(Normal, Basic, Psychic, 110, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2069, u8"ゲノムチャージ", u8"自分の山札から「基本【超】エネルギー」を2枚まで選び、このポケモンにつける。そして山札を切る。", "", { Colorless })
		.textEn(u8"Genome Charge", u8"Search your deck for up to 2 Basic {P} Energy cards and attach them to this Pokémon. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckAttachEnergyMeAndShuffle(2).targetCardId(PSYCHIC_ENERGY)
		.attack(2070, u8"サイコキネシス", u8"相手のバトルポケモンについているエネルギーの数×20ダメージ追加。", "80+", { Psychic, Psychic, Colorless })
		.textEn(u8"Psychic", u8"This attack does 20 more damage for each Energy attached to your opponent’s Active Pokémon.")
		.preEffect(AttackDamageChangeEnergyCount, Enemy).eVal(20).targetActive();

	CreateCard(1559, u8"デオキシス", Pokemon, 1559)
		.nameEn(u8"Deoxys")
		.pokemon(Normal, Basic, Psychic, 120, 2)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2071, u8"サイコスピア", u8"このワザを使うためのエネルギーより、2個多くエネルギーがついているなら、相手のベンチポケモン1匹にも、120ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "120", { Psychic, Psychic, Psychic })
		.textEn(u8"Psyspear", u8"If this Pokémon has at least 2 extra Energy attached (in addition to this attack’s cost), this attack also does 120 damage to 1 of your opponent’s Benched Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.setPostEffect()
		.condition(ConditionType::AttackEnergyExtra, 2, GreaterEqual)
		.postEffectDamageBench(120);

	CreateCard(1560, u8"デオキシス", Pokemon, 1560)
		.nameEn(u8"Deoxys")
		.pokemon(Normal, Basic, Psychic, 130, 3)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2072, u8"サイコプロテクト", u8"次の相手の番、このポケモンは特性を持つポケモンからワザのダメージを受けない。", "80", { Psychic, Psychic, Colorless })
		.textEn(u8"Psy Protection", u8"During your opponent’s next turn, prevent all damage done to this Pokémon by attacks from Pokémon that have an Ability.")
		.postEffectMe(NoDamageAbilityAttackNextEnemyTurn);

	CreateCard(1561, u8"デオキシス", Pokemon, 1561)
		.nameEn(u8"Deoxys")
		.pokemon(Normal, Basic, Psychic, 100, 0)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2073, u8"サイコスピード", u8"のぞむなら、自分の手札が5枚になるように、山札を引く。", "30", { Psychic })
		.textEn(u8"Psyspeed", u8"You may draw cards until you have 5 cards in your hand.")
		.setPostEffect()
		.conditionLess(5).targetHand()
		.postEffectSelectActivate()
		.effectDrawUntil(5);

	CreateCard(1562, u8"メガフラエッテex", Pokemon, 1562)
		.nameEn(u8"Mega Floette ex")
		.pokemon(MegaEx, Basic, Psychic, 250, 1)
		.weakness(Metal)
		.attack(2074, u8"やさしいひかり", u8"おたがいのポケモン全員のHPを、それぞれ「30」回復する。", "", { Psychic })
		.textEn(u8"Gentle Light", u8"Heal 30 damage from each Pokémon (both yours and your opponent’s).")
		.postEffect(Heal, Both).eVal(30).targetPokemon()
		.attack(2075, u8"エタニティブルーム", u8"自分の山札から「基本【超】エネルギー」を4枚まで選び、ベンチポケモンに好きなようにつける。そして山札を切る。", "200", { Psychic, Psychic, Psychic })
		.textEn(u8"Eternity Bloom", u8"Search your deck for up to 4 Basic {P} Energy cards and attach them to your Benched Pokémon in any way you like. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckAttachEnergyBenchAndShuffle(4).targetCardId(PSYCHIC_ENERGY);

	CreateCard(1563, u8"ニャスパー", Pokemon, 1563)
		.nameEn(u8"Espurr")
		.pokemon(Normal, Basic, Psychic, 60, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2076, u8"バディアタック", u8"この番に、手札から「マチエール」を出して使っていたなら、60ダメージ追加。", "10+", { Psychic })
		.textEn(u8"Buddy Attack", u8"If you played Emma from your hand during this turn, this attack does 60 more damage.")
		.setPreEffect()
		.exist(AreaType::TurnPlay, Me).targetName(u8"マチエール")
		.preEffectAttackDamageChange(60);

	CreateCard(1564, u8"ニャオニクス", Pokemon, 1564)
		.nameEn(u8"Meowstic")
		.pokemon(Normal, Stage1, Psychic, 100, 1)
		.evolvesFrom(u8"ニャスパー")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2077, u8"トリックステップ", u8"のぞむなら、相手のバトルポケモンについているエネルギーを1個選び、相手のベンチポケモンにつけ替える。", "80", { Psychic, Colorless })
		.textEn(u8"Tricky Steps", u8"You may move an Energy from your opponent’s Active Pokémon to 1 of their Benched Pokémon.")
		.setPostEffect()
		.exist(AreaType::Energy, Enemy).targetCondition(TargetType::AttachedActivePokemon)
		.existEnemyBench()
		.postEffectSelectActivate()
		.effect(SelectSwitchEnergy, Enemy).targetAttachedEnergy().targetCondition(TargetType::AttachedActivePokemon).selectEnergy(1)
		.effect(EnergySwitchEach, Enemy).eachSelectedList().targetBench().singleSelect();

	// Spiteful Evolution is Exeggcute's self-evolve (SelectEvolvesFrom on this card,
	// then EvolvesToEach) drawing from the hand instead of the deck; the counters go
	// on the card that was put on top, which is what the evolve leaves as Effected.
	CreateCard(1565, u8"ボクレー", Pokemon, 1565)
		.nameEn(u8"Phantump")
		.pokemon(Normal, Basic, Psychic, 70, 2)
		.weakness(Darkness)
		.resistance(Fighting)
		.activateSkillOnceTurn(525, u8"うらみしんか", u8"自分の番に1回使える。このポケモンから進化するカードを、自分の手札から1枚選び、このポケモンにのせて進化させる。その後、進化させたポケモンに、ダメカンを2個のせる。（最初の自分の番には使えない。）")
		.textEn(u8" Spiteful Evolution", u8"Once during your turn, you may use this Ability. Choose a card in your hand that evolves from this Pokémon and put it onto this Pokémon to evolve it. If you do, place 2 damage counters on the Pokémon you evolved in this way. You can’t use this Ability during your first turn.")
		.conditionNotFirstTurn()
		.exist(AreaType::Hand).targetCondition(TargetType::CanEvolveMe)
		.effectMe(SelectEvolvesFrom)
		.effectEvolvesToEach().targetHand()
		.separator()
		.effectEffectedCard(DamageCounter).eVal(2)
		.attack(2078, u8"つぶやく", u8"", "10", { Psychic })
		.textEn(u8"Mumble", u8"");

	CreateCard(1566, u8"オーロット", Pokemon, 1566)
		.nameEn(u8"Trevenant")
		.pokemon(Normal, Stage1, Psychic, 130, 3)
		.evolvesFrom(u8"ボクレー")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2079, u8"のろいのねっこ", u8"次の相手の番、このワザを受けたポケモンは、手札から出すエネルギーをつけられない。", "30", { Psychic })
		.textEn(u8"Cursed Roots", u8"During your opponent’s next turn, Energy can’t be attached from your opponent’s hand to the Defending Pokémon.")
		.postEffectActiveEnemy(CannotHandAttachEnergyNextTurn)
		.attack(2080, u8"オーバーペイン", u8"相手のポケモン全員にのっているダメカンの数×10ダメージ追加。", "60+", { Psychic, Psychic })
		.textEn(u8"Overwhelming Pain", u8"This attack does 10 more damage for each damage counter on all of your opponent’s Pokémon.")
		.preEffect(AttackDamageChangeDamageCounter, Enemy).eVal(10).targetPokemon();

	CreateCard(1567, u8"バケッチャ", Pokemon, 1567)
		.nameEn(u8"Pumpkaboo")
		.pokemon(Normal, Basic, Psychic, 60, 2)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2081, u8"ふむ", u8"", "20", { Psychic })
		.textEn(u8"Stampede", u8"");

	CreateCard(1568, u8"パンプジンex", Pokemon, 1568)
		.nameEn(u8"Gourgeist ex")
		.pokemon(Ex, Stage1, Psychic, 270, 2)
		.evolvesFrom(u8"バケッチャ")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2082, u8"ホラーロンド", u8"ダメカンがのっている自分のベンチポケモンの数×50ダメージ追加。", "30+", { Psychic })
		.textEn(u8"Horrifying Rondo", u8"This attack does 50 more damage for each of your Benched Pokémon that has any damage counters on it.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(50).targetBench().targetDamaged()
		.attack(2083, u8"ゴーストタッチ", u8"相手の手札からオモテを見ないで1枚選び、トラッシュする。", "140", { Psychic, Psychic })
		.textEn(u8"Ghostly Touch", u8"Discard a random card from your opponent’s hand.")
		.postEffect(ToTrash, Enemy).targetHand().randomSelect();

	CreateCard(1569, u8"ゼルネアス", Pokemon, 1569)
		.nameEn(u8"Xerneas")
		.pokemon(Normal, Basic, Psychic, 130, 2)
		.weakness(Metal)
		.attack(2084, u8"ジオストーム", u8"自分のポケモン全員についている【超】エネルギーの数×30ダメージ。", "30×", { Psychic, Psychic, Psychic })
		.textEn(u8"Geo Storm", u8"This attack does 30 damage for each {P} Energy attached to all of your Pokémon.")
		.preEffect(AttackDamageChangeTypeEnergyCount, Me).eVal((int)EnergyType::Psychic, 30).targetPokemon();

	CreateCard(1570, u8"ウソッキー", Pokemon, 1570)
		.nameEn(u8"Sudowoodo")
		.pokemon(Normal, Basic, Fighting, 110, 1)
		.weakness(Grass)
		.attack(2085, u8"しれんのたび", u8"自分の山札から「変化の書」を2枚まで選び、相手に見せて、手札に加える。そして山札を切る。", "", { Colorless })
		.textEn(u8"Trials and Trip-ulations", u8"Search your deck for up to 2 Transformation Tome cards, reveal them, and put them into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToHandAndShuffle(2).targetName(u8"変化の書")
		.attack(2086, u8"いわとばし", u8"このワザのダメージは抵抗力を計算しない。", "30", { Fighting })
		.textEn(u8"Rock Hurl", u8"This attack’s damage isn’t affected by Resistance.")
		.noTargetResistance();

	CreateCard(1571, u8"ゴマゾウ", Pokemon, 1571)
		.nameEn(u8"Phanpy")
		.pokemon(Normal, Basic, Fighting, 80, 2)
		.weakness(Grass)
		.attack(2087, u8"どろかけ", u8"", "10", { Fighting })
		.textEn(u8"Mud-Slap", u8"")
		.attack(2088, u8"ころがる", u8"", "40", { Colorless, Colorless, Colorless })
		.textEn(u8"Rollout", u8"");

	CreateCard(1572, u8"ドンファン", Pokemon, 1572)
		.nameEn(u8"Donphan")
		.pokemon(Normal, Stage1, Fighting, 150, 3)
		.evolvesFrom(u8"ゴマゾウ")
		.weakness(Grass)
		.attack(2089, u8"たたみかける", u8"次の自分の番、このポケモンが使うワザの、相手のバトルポケモンへのダメージは「+120」される。", "20", { Fighting })
		.textEn(u8"No Reprieve", u8"During your next turn, attacks used by this Pokémon do 120 more damage to your opponent’s Active Pokémon (before applying Weakness and Resistance).")
		.postEffectMe(DamageChangeActiveNextTurn).eVal(120)
		.attack(2090, u8"スマッシュヘッド", u8"このポケモンについているエネルギーを2個選び、トラッシュする。", "180", { Fighting, Colorless, Colorless, Colorless })
		.textEn(u8"Smashing Headbutt", u8"Discard 2 Energy from this Pokémon.")
		.postEffectTrashEnergyMe(2);

	CreateCard(1573, u8"ヤジロン", Pokemon, 1573)
		.nameEn(u8"Baltoy")
		.pokemon(Normal, Basic, Fighting, 70, 2)
		.weakness(Grass)
		.attack(2091, u8"れんぞくスピン", u8"ウラが出るまでコインを投げ、オモテの数×30ダメージ。", "30×", { Fighting })
		.textEn(u8"Continuous Spin", u8"Flip a coin until you get tails. This attack does 30 damage for each heads.")
		.preEffect(AttackDamageChangeCoinUntilTail, None).eVal(30);

	CreateCard(1574, u8"ネンドール", Pokemon, 1574)
		.nameEn(u8"Claydol")
		.pokemon(Normal, Stage1, Fighting, 120, 2)
		.evolvesFrom(u8"ヤジロン")
		.weakness(Grass)
		.attack(2092, u8"たいかこうせん", u8"相手の進化しているバトルポケモンから、「進化カード」を1枚はがして退化させる。はがしたカードは、相手の手札にもどす。", "50", { Fighting })
		.textEn(u8"Devolution Ray", u8"If your opponent’s Active Pokémon is an evolved Pokémon, devolve it by putting the highest Stage Evolution card on it into your opponent’s hand.")
		.setPostEffect()
		.exist(AreaType::Active, Enemy).targetCondition(TargetType::Evolved)
		.effect(Devolve, Enemy).eVal((int)AreaType::Hand).targetActive().targetCondition(TargetType::Evolved);

	CreateCard(1575, u8"メガエルレイドex", Pokemon, 1575)
		.nameEn(u8"Mega Gallade ex")
		.pokemon(MegaEx, Stage2, Fighting, 350, 2)
		.evolvesFrom(u8"キルリア")
		.weakness(Psychic)
		.attack(2093, u8"はやてぎり", u8"このポケモンにダメカンがのっていないなら、150ダメージ追加。", "50+", { Fighting })
		.textEn(u8"Gale Slash", u8"If this Pokémon has no damage counters on it, this attack does 150 more damage.")
		.setPreEffect()
		.exist(AreaType::Me).targetNotDamaged()
		.preEffectAttackDamageChange(150)
		.attack(2094, u8"マーベラスエッジ", u8"", "240", { Fighting, Fighting, Colorless })
		.textEn(u8"Marvelous Edge", u8"");

	CreateCard(1576, u8"ズバット", Pokemon, 1576)
		.nameEn(u8"Zubat")
		.pokemon(Normal, Basic, Darkness, 40, 0)
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2095, u8"ちょうおんぱ", u8"相手のバトルポケモンを【こんらん】にする。", "", { Darkness })
		.textEn(u8"Supersonic", u8"Your opponent’s Active Pokémon is now Confused.")
		.postEffect(Confuse, Enemy);

	CreateCard(1577, u8"ゴルバット", Pokemon, 1577)
		.nameEn(u8"Golbat")
		.pokemon(Normal, Stage1, Darkness, 80, 1)
		.evolvesFrom(u8"ズバット")
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2096, u8"おんみつひこう", u8"次の相手の番、このポケモンは【たね】ポケモンからワザのダメージを受けない。", "30", { Darkness })
		.textEn(u8"Covert Flight", u8"During your opponent’s next turn, prevent all damage done to this Pokémon by attacks from Basic Pokémon.")
		.postEffectMe(NoDamageBasicAttackNextEnemyTurn);

	// Nighttime Maneuvers is Dialga's Time Manipulation with one card.
	CreateCard(1578, u8"クロバット", Pokemon, 1578)
		.nameEn(u8"Crobat")
		.pokemon(Normal, Stage2, Darkness, 130, 1)
		.evolvesFrom(u8"ゴルバット")
		.weakness(Lightning)
		.resistance(Fighting)
		.activateSkillOnceTurnActive(526, u8"よるこうさく", u8"このポケモンがバトル場にいるなら、自分の番に1回使える。自分の山札から好きなカードを1枚選ぶ。残りの山札を切り、選んだカードを山札の上にもどす。")
		.textEn(u8" Nighttime Maneuvers", u8"Once during your turn, if this Pokémon is in the Active Spot, you may use this Ability. Search your deck for a card. Shuffle your deck, then put that card on top of it.")
		.existMyDeck()
		.effect(ToLooking, Me).targetDeck().singleSelect().setContext(SelectContext::ToDeck)
		.effectShuffle()
		.effect(ToDeckReverse, Me).targetLooking()
		.attack(2097, u8"どくおんぱ", u8"相手のバトルポケモンを【どく】と【こんらん】にする。", "80", { Darkness })
		.textEn(u8"Poison Sound Wave", u8"Your opponent’s Active Pokémon is now Confused and Poisoned.")
		.postEffect(Confuse, Enemy)
		.effect(Poison, Enemy);

	CreateCard(1579, u8"ハリーセン", Pokemon, 1579)
		.nameEn(u8"Qwilfish")
		.pokemon(Normal, Basic, Darkness, 90, 1)
		.weakness(Fighting)
		.abilityActive(527, u8"どくのトゲ", u8"このポケモンが、バトル場で相手のポケモンからワザのダメージを受けたとき、ワザを使ったポケモンを【どく】にする。")
		.textEn(u8" Poison Point", u8"If this Pokémon is in the Active Spot and is damaged by an attack from your opponent’s Pokémon (even if this Pokémon is Knocked Out), the Attacking Pokémon is now Poisoned.")
		.triggerMe(TriggerType::DamagedEnemyAttackActive)
		.effectTriggerObject(Poison)
		.attack(2098, u8"ベノムショック", u8"相手のバトルポケモンが【どく】なら、50ダメージ追加。", "30+", { Darkness })
		.textEn(u8"Venoshock", u8"If your opponent’s Active Pokémon is Poisoned, this attack does 50 more damage.")
		.setPreEffect()
		.exist(AreaType::Active, Enemy).targetCondition(TargetType::Poison)
		.preEffectAttackDamageChange(50);

	CreateCard(1580, u8"スカンプー", Pokemon, 1580)
		.nameEn(u8"Stunky")
		.pokemon(Normal, Basic, Darkness, 70, 2)
		.weakness(Fighting)
		.attack(2099, u8"ひっかく", u8"", "20", { Darkness })
		.textEn(u8"Scratch", u8"");

	CreateCard(1581, u8"スカタンク", Pokemon, 1581)
		.nameEn(u8"Skuntank")
		.pokemon(Normal, Stage1, Darkness, 110, 2)
		.evolvesFrom(u8"スカンプー")
		.weakness(Fighting)
		.attack(2100, u8"うしろげり", u8"", "40", { Darkness })
		.textEn(u8"Rear Kick", u8"")
		.attack(2101, u8"スマッシュターン", u8"このポケモンをベンチポケモンと入れ替える。", "100", { Darkness, Darkness, Colorless })
		.textEn(u8"Smash Turn", u8"Switch this Pokémon with 1 of your Benched Pokémon.")
		.setPostEffect()
		.effectSwitch(Me);

	CreateCard(1582, u8"ワルビアルex", Pokemon, 1582)
		.nameEn(u8"Krookodile ex")
		.pokemon(Ex, Stage2, Darkness, 320, 3)
		.evolvesFrom(u8"ワルビル")
		.weakness(Grass)
		.attack(2102, u8"おいつめる", u8"次の相手の番、このワザを受けたポケモンは、にげられない。", "80", { Darkness, Colorless })
		.textEn(u8"Corner", u8"During your opponent’s next turn, the Defending Pokémon can’t retreat.")
		.postEffectActiveEnemy(CannotRetreatNextTurn)
		.attack(2103, u8"ストロングバイト", u8"このポケモンに「ポケモンのどうぐ」がついているなら、140ダメージ追加。", "140+", { Darkness, Darkness, Colorless })
		.textEn(u8"Strong Bite", u8"If this Pokémon has a Pokémon Tool attached, this attack does 140 more damage.")
		.setPreEffect()
		.exist(AreaType::Me).targetCondition(TargetType::IsAttachedTool)
		.preEffectAttackDamageChange(140);

	CreateCard(1583, u8"ヤブクロン", Pokemon, 1583)
		.nameEn(u8"Trubbish")
		.pokemon(Normal, Basic, Darkness, 70, 2)
		.weakness(Fighting)
		.attack(2104, u8"アシッドボム", u8"コインを1回投げオモテなら、相手のバトルポケモンについているエネルギーを1個選び、トラッシュする。", "10", { Darkness })
		.textEn(u8"Acid Spray", u8"Flip a coin. If heads, discard an Energy from your opponent’s Active Pokémon.")
		.postEffectActiveEnemyEnergyTrashIfCoinHead();

	CreateCard(1584, u8"ダストダス", Pokemon, 1584)
		.nameEn(u8"Garbodor")
		.pokemon(Normal, Stage1, Darkness, 140, 3)
		.evolvesFrom(u8"ヤブクロン")
		.weakness(Fighting)
		.abilityBattleField(528, u8"ゴミダウナー", u8"このポケモンがいるかぎり、「ポケモンのどうぐ」がついている相手のバトルポケモンが使うワザのダメージは「-20」される。")
		.textEn(u8" Gloomy Garbage", u8"Attacks used by your opponent’s Active Pokémon that has a Pokémon Tool attached do 20 less damage (before applying Weakness and Resistance).")
		.effect(DamageChange, Enemy).eVal(-20).targetActive().targetCondition(TargetType::IsAttachedTool)
		.attack(2105, u8"ヘドロばくだん", u8"", "100", { Darkness, Darkness, Colorless })
		.textEn(u8"Sludge Bomb", u8"");

	CreateCard(1585, u8"クズモー", Pokemon, 1585)
		.nameEn(u8"Skrelp")
		.pokemon(Normal, Basic, Darkness, 70, 1)
		.weakness(Fighting)
		.attack(2106, u8"ひっかける", u8"", "10", { Colorless })
		.textEn(u8"Hook", u8"");

	CreateCard(1586, u8"テッシード", Pokemon, 1586)
		.nameEn(u8"Ferroseed")
		.pokemon(Normal, Basic, Metal, 70, 2)
		.weakness(Fire)
		.resistance(Grass)
		.attack(2107, u8"ころがりタックル", u8"", "40", { Metal, Metal })
		.textEn(u8"Rolling Tackle", u8"");

	// Startling Drop rides TriggerType::DeckToTrashEnemyEffect, which CardMove.h
	// already pulls for a deck-to-discard move caused by the other player's Pokémon,
	// Item or Supporter. By then the card is in the discard pile, which PullTrigger
	// scans, so the ability lives there (abilityTrash).
	CreateCard(1587, u8"ナットレイ", Pokemon, 1587)
		.nameEn(u8"Ferrothorn")
		.pokemon(Normal, Stage1, Metal, 130, 3)
		.evolvesFrom(u8"テッシード")
		.weakness(Fire)
		.resistance(Grass)
		.abilityTrash(529, u8"どっきりおとし", u8"相手の番に、このカードが相手のワザ・特性・グッズ・サポートの効果で山札からトラッシュされたとき、相手の山札を上から8枚トラッシュする。")
		.textEn(u8" Startling Drop", u8"During your opponent’s turn, if this Pokémon is discarded from your deck by an effect of an attack or Ability from your opponent’s Pokémon, or by an effect of your opponent’s Item or Supporter cards, discard the top 8 cards of your opponent’s deck.")
		.triggerMe(TriggerType::DeckToTrashEnemyEffect)
		.conditionEnemyTurn()
		.effect(DeckToTrash, Enemy).eVal(8)
		.attack(2108, u8"スペシャルウィップ", u8"このポケモンに特殊エネルギーがついているなら、70ダメージ追加。", "70+", { Metal, Metal })
		.textEn(u8"Special Whip", u8"If this Pokémon has any Special Energy attached, this attack does 70 more damage.")
		.setPreEffect()
		.exist(AreaType::Me).targetCondition(TargetType::IsAttachedSpecialEnergy)
		.preEffectAttackDamageChange(70);

	CreateCard(1588, u8"コバルオンex", Pokemon, 1588)
		.nameEn(u8"Cobalion ex")
		.pokemon(Ex, Basic, Metal, 210, 2)
		.weakness(Fire)
		.resistance(Grass)
		.abilityBenchToActive(530, u8"メタルロード", u8"自分の番に、このポケモンがベンチからバトル場に出たとき、1回使える。自分の場のポケモンについている【鋼】エネルギーを好きなだけ選び、このポケモンにつけ替える。")
		.textEn(u8" Metal Road", u8"Once during your turn, when this Pokémon moves from your Bench to the Active Spot, you may use this Ability. Move any amount of {M} Energy from your other Pokémon to this Pokémon.")
		.effect(SelectSwitchEnergyCard, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedBenchPokemon).targetEnergyTypeAttached(Metal).selectEnergyAny()
		.effect(EnergySwitchEach, Me).targetActive().eachSelectedList()
		.attack(2109, u8"パワータックル", u8"次の自分の番、このポケモンはワザが使えない。", "200", { Metal, Metal, Colorless })
		.textEn(u8"Power Tackle", u8"During your next turn, this Pokémon can’t use attacks.")
		.postEffectMe(CannotAttackNextTurn);

	CreateCard(1589, u8"メガドラミドロex", Pokemon, 1589)
		.nameEn(u8"Mega Dragalge ex")
		.pokemon(MegaEx, Stage1, Dragon, 330, 2)
		.evolvesFrom(u8"クズモー")
		.attack(2110, u8"ふしょくえき", u8"相手のポケモン全員についている「ポケモンのどうぐ」と「特殊エネルギー」を、すべてトラッシュする。", "", { Colorless, Colorless })
		.textEn(u8"Corrosive Liquid", u8"Discard all Pokémon Tools and Special Energy from all of your opponent’s Pokémon.")
		.postEffect(ToTrash, Enemy).targetAttachedTool()
		.effect(ToTrash, Enemy).targetAttachedEnergy().targetSpecialEnergy()
		.attack(2111, u8"デッドリーポイズン", u8"相手のバトルポケモンを【どく】にする。この【どく】でのせるダメカンの数は16個になる。", "", { Water, Darkness })
		.textEn(u8"Pernicious Poison", u8"Your opponent’s Active Pokémon is now Poisoned. During Pokémon Checkup, place 16 damage counters on that Pokémon instead of 1.")
		.postEffect(Poison16, Enemy);

	CreateCard(1590, u8"ヌメラ", Pokemon, 1590)
		.nameEn(u8"Goomy")
		.pokemon(Normal, Basic, Dragon, 60, 2)
		.attack(2112, u8"すいとる", u8"このポケモンのHPを「30」回復する。", "30", { Water, Psychic })
		.textEn(u8"Absorb", u8"Heal 30 damage from this Pokémon.")
		.postEffectMe(Heal).eVal(30);

	CreateCard(1591, u8"ヌメイル", Pokemon, 1591)
		.nameEn(u8"Sliggoo")
		.pokemon(Normal, Stage1, Dragon, 90, 3)
		.evolvesFrom(u8"ヌメラ")
		.attack(2113, u8"ひっぱたく", u8"", "70", { Water, Psychic })
		.textEn(u8"Gentle Slap", u8"");

	// Slimy Sliding: PreRetreat is pulled before the cost is paid, and FailRetreat makes
	// SelectedRetreat2 return before discarding or switching -- the tails outcome.
	// The engine flips for the ability's owner; the odds are the same either way.
	CreateCard(1592, u8"ヌメルゴン", Pokemon, 1592)
		.nameEn(u8"Goodra")
		.pokemon(Normal, Stage2, Dragon, 160, 3)
		.evolvesFrom(u8"ヌメイル")
		.abilityBattleField(531, u8"ぬめぬめスリップ", u8"このポケモンがいるかぎり、相手のバトルポケモンがにげるとき、相手はコインを1回投げる。ウラなら、にげるためのエネルギーはトラッシュせず、入れ替えをしない。この特性の効果は重ならない。")
		.textEn(u8" Slimy Sliding", u8"When your opponent’s Active Pokémon retreats, your opponent flips a coin. If tails, Energy for its Retreat Cost is not discarded, and they don’t switch Pokémon. The effect of Slimy Sliding doesn’t stack.")
		.trigger(TriggerType::PreRetreat).targetActive().targetPlayer(Enemy).notStack()
		.effect(BreakIfCoinHead, None)
		.effect(FailRetreat, None)
		.attack(2114, u8"りゅうのはどう", u8"自分の山札を上から1枚トラッシュする。", "160", { Water, Psychic })
		.textEn(u8"Dragon Pulse", u8"Discard the top card of your deck.")
		.postEffect(DeckToTrash, Me).eVal(1);

	CreateCard(1593, u8"ミネズミ", Pokemon, 1593)
		.nameEn(u8"Patrat")
		.pokemon(Normal, Basic, Colorless, 70, 1)
		.weakness(Fighting)
		.abilityBattleField(532, u8"かんしのめ", u8"このポケモンがいるかぎり、おたがいのポケモン全員にのっているダメカンは、別のポケモンにのせ替えられない。")
		.textEn(u8" Watchful Eye", u8"Damage counters on each Pokémon (both yours and your opponent’s) can’t be moved to other Pokémon.")
		.effect(CannotMoveDamageCounter, Both).targetPokemon()
		.attack(2115, u8"かみつく", u8"", "10", { Colorless })
		.textEn(u8"Bite", u8"");

	// Snap Inspection: 3 coins, and on no heads the CoinHeadCount condition stops
	// before anything is revealed; the pick then takes exactly that many heads.
	CreateCard(1594, u8"ミルホッグ", Pokemon, 1594)
		.nameEn(u8"Watchog")
		.pokemon(Normal, Stage1, Colorless, 100, 1)
		.evolvesFrom(u8"ミネズミ")
		.weakness(Fighting)
		.attack(2116, u8"ぬきうちチェック", u8"コインを3回投げる。オモテが出たなら、相手の手札を見て、その中からカードをオモテの数ぶん選び、相手の山札にもどして切る。", "", { Colorless })
		.textEn(u8"Snap Inspection", u8"Flip 3 coins. If any of them are heads, your opponent reveals their hand. For each heads, choose a card you find there and shuffle it into your opponent’s deck.")
		.setPostEffect()
		.exist(AreaType::Hand, Enemy)
		.effect(Coin, None).eVal(3)
		.condition(ConditionType::CoinHeadCount, 1, GreaterEqual)
		.effect(ToLooking, Enemy).targetHand()
		.effect(ToDeckAndShuffle, Enemy).targetLooking().selectCoinHeadCount()
		.effect(ToHand, Enemy).targetLooking()
		.attack(2117, u8"けたぐり", u8"", "50", { Colorless })
		.textEn(u8"Low Kick", u8"");

	CreateCard(1595, u8"チラーミィ", Pokemon, 1595)
		.nameEn(u8"Minccino")
		.pokemon(Normal, Basic, Colorless, 70, 1)
		.weakness(Fighting)
		.attack(2118, u8"とっしん", u8"このポケモンにも10ダメージ。", "30", { Colorless })
		.textEn(u8"Take Down", u8"This Pokémon also does 10 damage to itself.")
		.postEffectDamageMe(10);

	CreateCard(1596, u8"チラチーノex", Pokemon, 1596)
		.nameEn(u8"Cinccino ex")
		.pokemon(Ex, Stage1, Colorless, 240, 1)
		.evolvesFrom(u8"チラーミィ")
		.weakness(Fighting)
		.abilityBattleField(533, u8"なめらかコート", u8"このポケモンがワザのダメージを受けるとき、自分はコインを1回投げる。オモテなら、このポケモンはそのダメージを受けない。")
		.textEn(u8" Smooth Coat", u8"If any damage is done to this Pokémon by attacks, flip a coin. If heads, prevent that damage.")
		.effectMe(NoDamageCoin)
		.attack(2119, u8"エナジービンタ", u8"このポケモンについているエネルギーの数×40ダメージ。", "40×", { Colorless })
		.textEn(u8"Energized Slap", u8"This attack does 40 damage for each Energy attached to this Pokémon.")
		.preEffectMe(AttackDamageChangeEnergyCount).eVal(40);

	// Ange Floette takes the id Core.h reserves for it (ANGE_FLOETTE): GameProc.h
	// already lets it be played only over Prism Tower, and on the same turn as one.
	CreateCard(1429, u8"アンジュフラエッテ", Stadium, 1429)
		.nameEn(u8"Ange Floette")
		.stadiumSkill(534, u8"このカードは、場に出ている「プリズムタワー」をトラッシュしなければ場に出せず、「プリズムタワー」を出した番でも場に出せる。おたがいの場の「メガフラエッテex」全員は、それぞれ最大HPが「+150」される。")
		.textEn(u8"Ange Floette", u8"You can put this card into play only if you discard a Prism Tower in play, and you can put this card into play during the same turn you play Prism Tower.\n\nEach Mega Floette ex in play (both yours and your opponent’s) gets +150 HP.")
		.effect(MaxHpChange, Both).eVal(150).targetPokemon().targetName(u8"メガフラエッテex");

	// Emma: Morty's Conviction's draw-per-counted-target over the revealed hand.
	CreateCard(1597, u8"マチエール", Supporter, 1597)
		.nameEn(u8"Emma")
		.playSkill(535, u8"相手の手札を見て、その中にあるポケモンの枚数ぶん、自分の山札を引く。")
		.textEn(u8"Emma", u8"Your opponent reveals their hand, and you draw a card for each Pokémon you find there.")
		.exist(AreaType::Hand, Enemy)
		.effect(ToLooking, Enemy).targetHand()
		.effect(NoEffect, Enemy).targetLooking().targetPokemonCard()
		.effectDraw(1).multiplyEffectValuePreTargetCount()
		.effect(ToHand, Enemy).targetLooking();

	CreateCard(1598, u8"大漁ネット", Item, 1598)
		.nameEn(u8"Great Haul Net")
		.playSkill(536, u8"自分のトラッシュから【水】ポケモンと「基本【水】エネルギー」をそれぞれ3枚まで選び、相手に見せて、山札にもどして切る。")
		.textEn(u8"Great Haul Net", u8"Choose 1 or both:\n• Shuffle up to 3 {W} Pokémon from your discard pile into your deck.\n• Shuffle up to 3 Basic {W} Energy cards from your discard pile into your deck.")
		.existAnyTargetAfterEffect()
		.effect(ToDeckAndShuffle, Me).targetTrash().targetPokemonCard().targetEnergyType(Water).maxSelect(3).canNoSelect()
		.effect(ToDeckAndShuffle, Me).targetTrash().targetCardId(WATER_ENERGY).maxSelect(3).canNoSelect();

	CreateCard(1599, u8"ジプソ", Supporter, 1599)
		.nameEn(u8"Philippe")
		.playSkill(537, u8"自分のトラッシュから「基本【鋼】エネルギー」を2枚まで選び、自分の【鋼】ポケモン1匹につける。")
		.textEn(u8"Philippe", u8"Attach up to 2 Basic {M} Energy cards from your discard pile to 1 of your {M} Pokémon.")
		.existPokemon().targetEnergyType(Metal)
		.effectSelectAttachBasicEnergyTrash(2).targetCardId(METAL_ENERGY)
		.effect(AttachSelectedCard, Me).targetPokemon().targetEnergyType(Metal).singleSelect();

	CreateCard(1600, u8"ホミカの演奏", Supporter, 1600)
		.nameEn(u8"Roxie’s Performance")
		.playSkill(538, u8"次の相手の番、相手の【どく】のポケモンは、にげられない。（新しく【どく】にしたポケモンもふくむ。）")
		.textEn(u8"Roxie’s Performance", u8"During your opponent’s next turn, their Poisoned Pokémon can’t retreat. (This includes newly Poisoned Pokémon.)")
		.effect(CannotRetreatPoison, Enemy);

	CreateCard(1601, u8"バブル水エネルギー", SpecialEnergy, 1601)
		.nameEn(u8"Bubbly Water Energy")
		.energySkill(539, u8"このカードは、ポケモンについているかぎり、【水】エネルギー1個ぶんとしてはたらく。このカードをつけている【水】ポケモンは、特殊状態にならず、受けている特殊状態は、すべて回復する。")
		.textEn(u8"Bubbly Water Energy", u8"As long as this card is attached to a Pokémon, it provides {W} Energy.\n\nThe {W} Pokémon this card is attached to recovers from all Special Conditions and can’t be affected by any Special Conditions.")
		.effectEnergyContinual(NoSpecialCondition).targetEnergyType(Water)
		.specialEnergy(EnergyType::Water, 1);

	// Nitro Fire Energy takes the id Core.h reserves for it (NITRO_FIRE_ENERGY), so
	// SelectProc.h's AfterEnergyDiscard queues this Attach trigger exactly as it does
	// for Boomerang Energy -- only for a {R} attacker -- and the card goes to the hand.
	CreateCard(1268, u8"ニトロ炎エネルギー", SpecialEnergy, 1268)
		.nameEn(u8"Nitro Fire Energy")
		.energySkill(540, u8"このカードは、ポケモンについているかぎり、【炎】エネルギー1個ぶんとしてはたらく。このカードをつけている【炎】ポケモンが使うワザの効果で、このカードがトラッシュされたなら、ワザのダメージや効果のあとに、手札にもどす。")
		.textEn(u8"Nitro Fire Energy", u8"As long as this card is attached to a Pokémon, it provides {R} Energy.\n\nIf this card is discarded by an effect of an attack used by the {R} Pokémon this card is attached to, put this card into your hand after attack damage and effects.")
		.canActivateTrash()
		.trigger(TriggerType::Attach)
		.effectMe(ToHand)
		.specialEnergy(EnergyType::Fire, 1);

	CreateCard(1602, u8"トロピウス", Pokemon, 1602)
		.nameEn(u8"Tropius")
		.pokemon(Normal, Basic, Grass, 110, 1)
		.weakness(Fire)
		.attack(2120, u8"かじつのかおり", u8"自分の山札を上から6枚見て、その中からポケモンを好きなだけ選び、相手に見せて、手札に加える。残りのカードは山札にもどして切る。", "", { Colorless })
		.textEn(u8"Fruity Aroma", u8"Look at the top 6 cards of your deck, and you may reveal any number of Pokémon you find there and put them into your hand. Shuffle the other cards back into your deck.")
		.setPostEffect()
		.existMyDeck()
		.effect(LookDeck, Me).eVal(6)
		.effect(ToHand, Me).targetLooking().selectAny().targetPokemonCard()
		.effect(ToDeckReverse, Me).targetLooking()
		.effectShuffle()
		.attack(2121, u8"ソーラービーム", u8"", "60", { Grass, Colorless })
		.textEn(u8"Solar Beam", u8"");

	CreateCard(1603, u8"アゴジムシ", Pokemon, 1603)
		.nameEn(u8"Grubbin")
		.pokemon(Normal, Basic, Grass, 70, 2)
		.weakness(Fire)
		.attack(2122, u8"いとをはく", u8"コインを1回投げオモテなら、相手のバトルポケモンを【マヒ】にする。", "10", { Colorless })
		.textEn(u8"String Shot", u8"Flip a coin. If heads, your opponent’s Active Pokémon is now Paralyzed.")
		.postEffectParalyzeIfCoinHead();

	CreateCard(1604, u8"カリキリ", Pokemon, 1604)
		.nameEn(u8"Fomantis")
		.pokemon(Normal, Basic, Grass, 70, 1)
		.weakness(Fire)
		.attack(2123, u8"とつげき", u8"このポケモンにも10ダメージ。", "30", { Grass })
		.textEn(u8"Reckless Charge", u8"This Pokémon also does 10 damage to itself.")
		.postEffectDamageMe(10);

	CreateCard(1605, u8"ラランテスex", Pokemon, 1605)
		.nameEn(u8"Lurantis ex")
		.pokemon(Ex, Stage1, Grass, 260, 1)
		.evolvesFrom(u8"カリキリ")
		.weakness(Fire)
		.attack(2124, u8"はつらつカッター", u8"この番に、このポケモンのHPを回復していたなら、200ダメージ追加。", "60+", { Grass })
		.textEn(u8"Lively Cutter", u8"If this Pokémon was healed during this turn, this attack does 200 more damage.")
		.setPreEffect()
		.exist(AreaType::Me).targetCondition(TargetType::HealThisTurn)
		.preEffectAttackDamageChange(200)
		.attack(2125, u8"リーフガード", u8"次の相手の番、このポケモンが受けるワザのダメージは「-50」される。", "140", { Grass, Colorless })
		.textEn(u8"Leaf Guard", u8"During your opponent’s next turn, this Pokémon takes 50 less damage from attacks (after applying Weakness and Resistance).")
		.postEffectMe(TakeDamageChangeNextEnemyTurn).eVal(-50);

	CreateCard(1606, u8"メガマフォクシーex", Pokemon, 1606)
		.nameEn(u8"Mega Delphox ex")
		.pokemon(MegaEx, Stage2, Fire, 350, 2)
		.evolvesFrom(u8"テールナー")
		.weakness(Water)
		.attack(2126, u8"トリックポータル", u8"自分の山札を上から9枚見て、その中からポケモンを好きなだけ選び、ベンチに出す。残りのカードは山札にもどして切る。", "", { Fire })
		.textEn(u8"Trick Portal", u8"Look at the top 9 cards of your deck, and you may put any number of Pokémon you find there onto your Bench. Shuffle the other cards back into your deck.")
		.setPostEffect()
		.effectLookAndToBenchRestDeckAndShuffle(9, Me).targetPokemonCard()
		.attack(2127, u8"あやしいともしび", u8"相手のバトルポケモンを【やけど】と【こんらん】にする。", "200", { Fire, Colorless, Colorless })
		.textEn(u8"Eerie Glow", u8"Your opponent’s Active Pokémon is now Burned and Confused.")
		.postEffect(Burn, Enemy)
		.effect(Confuse, Enemy);

	// Bug Out counts the revealed Pokémon that have Bug Out (HasAttackName, the
	// match Round and United Wings use), then sends every revealed Pokémon back and
	// discards the rest.
	CreateCard(1607, u8"ヤクデ", Pokemon, 1607)
		.nameEn(u8"Sizzlipede")
		.pokemon(Normal, Basic, Fire, 80, 2)
		.weakness(Water)
		.attack(2128, u8"のやき", u8"相手の山札を上から1枚トラッシュする。", "", { Fire })
		.textEn(u8"Controlled Burn", u8"Discard the top card of your opponent’s deck.")
		.postEffectDeckToTrash(1)
		.attack(2129, u8"バグパニック", u8"自分の山札を下から7枚オモテにして、その中にある、ワザ「バグパニック」を持つポケモンの枚数×50ダメージ。オモテにしたポケモンは山札にもどして切る。残りのカードはトラッシュする。", "50×", { Colorless, Colorless, Colorless })
		.textEn(u8"Bug Out", u8"Reveal the bottom 7 cards of your deck, and this attack does 50 damage for each Pokémon you find there that has the Bug Out attack. Then, shuffle any revealed Pokémon back into your deck. Discard the other cards.")
		.setPreEffect()
		.existMyDeck()
		.effect(LookDeckBottom, Me).eVal(7)
		.effect(AttackDamageChangeTargetCount, Me).eVal(50).targetLooking().targetNameCondition(TargetType::HasAttackName, u8"バグパニック")
		.effect(ToDeck, Me).targetLooking().targetPokemonCard()
		.effect(ToTrash, Me).targetLooking()
		.effectShuffle();

	CreateCard(1608, u8"マルヤクデ", Pokemon, 1608)
		.nameEn(u8"Centiskorch")
		.pokemon(Normal, Stage1, Fire, 140, 3)
		.evolvesFrom(u8"ヤクデ")
		.weakness(Water)
		.attack(2130, u8"のやき", u8"相手の山札を上から2枚トラッシュする。", "", { Fire })
		.textEn(u8"Controlled Burn", u8"Discard the top 2 cards of your opponent’s deck.")
		.postEffectDeckToTrash(2)
		.attack(2131, u8"ヒートタックル", u8"このポケモンにも30ダメージ。", "160", { Fire, Colorless, Colorless, Colorless })
		.textEn(u8"Heat Tackle", u8"This Pokémon also does 30 damage to itself.")
		.postEffectDamageMe(30);

	CreateCard(1609, u8"カルボウ", Pokemon, 1609)
		.nameEn(u8"Charcadet")
		.pokemon(Normal, Basic, Fire, 80, 2)
		.weakness(Water)
		.attack(2132, u8"ぜんりょくパンチ", u8"コインを1回投げウラなら、このワザは失敗。", "40", { Fire })
		.textEn(u8"Best Punch", u8"Flip a coin. If tails, this attack does nothing.")
		.preEffectFailAttackCoinTail();

	CreateCard(1610, u8"グレンアルマ", Pokemon, 1610)
		.nameEn(u8"Armarouge")
		.pokemon(Normal, Stage1, Fire, 140, 2)
		.evolvesFrom(u8"カルボウ")
		.weakness(Water)
		.attack(2133, u8"フレイムレギオン", u8"【炎】エネルギーがついている自分のベンチポケモンの数×40ダメージ追加。", "40+", { Fire })
		.textEn(u8"Flame Legion", u8"This attack does 40 more damage for each of your Benched Pokémon that has any {R} Energy attached.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(40).targetBench().targetCondition(TargetType::IsAttachedEnergyType, (int)Fire);

	CreateCard(1611, u8"トサキント", Pokemon, 1611)
		.nameEn(u8"Goldeen")
		.pokemon(Normal, Basic, Water, 70, 1)
		.weakness(Lightning)
		.attack(2134, u8"つきさす", u8"", "30", { Colorless, Colorless })
		.textEn(u8"Pierce", u8"");

	CreateCard(1612, u8"アズマオウ", Pokemon, 1612)
		.nameEn(u8"Seaking")
		.pokemon(Normal, Stage1, Water, 110, 1)
		.evolvesFrom(u8"トサキント")
		.weakness(Lightning)
		.attack(2135, u8"ハイドロショット", u8"相手のポケモン1匹に、このポケモンについている【水】エネルギーの数×30ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Colorless, Colorless, Colorless })
		.textEn(u8"Hydro Jet", u8"This attack does 30 damage to 1 of your opponent’s Pokémon for each {W} Energy attached to this Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffectMe(EffectDamageChangeTypeEnergyCount).eVal((int)EnergyType::Water, 30)
		.postEffectDamagePokemon(0);

	CreateCard(1613, u8"ホエルコ", Pokemon, 1613)
		.nameEn(u8"Wailmer")
		.pokemon(Normal, Basic, Water, 130, 4)
		.weakness(Lightning)
		.attack(2136, u8"みずでっぽう", u8"", "40", { Water, Water })
		.textEn(u8"Water Gun", u8"")
		.attack(2137, u8"スプラッシュ", u8"", "80", { Water, Water, Water })
		.textEn(u8"Wave Splash", u8"");

	CreateCard(1614, u8"ホエルオーex", Pokemon, 1614)
		.nameEn(u8"Wailord ex")
		.pokemon(Ex, Stage1, Water, 380, 4)
		.evolvesFrom(u8"ホエルコ")
		.weakness(Lightning)
		.attack(2138, u8"なみのり", u8"", "120", { Water, Water, Water })
		.textEn(u8"Surf", u8"")
		.attack(2139, u8"フォーリングダウン", u8"このポケモンを【ねむり】にする。", "270", { Water, Water, Water, Water, Water })
		.textEn(u8"Falling Down", u8"This Pokémon is now Asleep.")
		.postEffect(Sleep, Me);

	CreateCard(1615, u8"ジーランス", Pokemon, 1615)
		.nameEn(u8"Relicanth")
		.pokemon(Normal, Basic, Water, 100, 1)
		.weakness(Lightning)
		.attack(2140, u8"フォッシルビート", u8"名前に「古びた」とつく自分のベンチポケモンの数×30ダメージ追加。", "10+", { Colorless })
		.textEn(u8"Fossil Beatdown", u8"This attack does 30 more damage for each of your Benched Pokémon that has “Antique” in its name.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(30).targetBench().targetNameContains(u8"古びた");

	CreateCard(1616, u8"アシマリ", Pokemon, 1616)
		.nameEn(u8"Popplio")
		.pokemon(Normal, Basic, Water, 70, 1)
		.weakness(Lightning)
		.attack(2141, u8"はたく", u8"", "20", { Water })
		.textEn(u8"Pound", u8"");

	CreateCard(1617, u8"オシャマリ", Pokemon, 1617)
		.nameEn(u8"Brionne")
		.pokemon(Normal, Stage1, Water, 90, 2)
		.evolvesFrom(u8"アシマリ")
		.weakness(Lightning)
		.attack(2142, u8"ハイパーボイス", u8"", "40", { Water })
		.textEn(u8"Hyper Voice", u8"");

	CreateCard(1618, u8"アシレーヌ", Pokemon, 1618)
		.nameEn(u8"Primarina")
		.pokemon(Normal, Stage2, Water, 150, 2)
		.evolvesFrom(u8"オシャマリ")
		.weakness(Lightning)
		.abilityEvolve(541, u8"まんたんメロディ", u8"自分の番に、このカードを手札から出して進化させたとき、1回使える。自分のポケモン1匹のHPを、すべて回復する。")
		.textEn(u8" Enriching Melody", u8"Once during your turn, when you play this Pokémon from your hand to evolve 1 of your Pokémon, you may use this Ability. Heal all damage from 1 of your Pokémon.")
		.existDamaged(Me)
		.effect(HealAll, Me).targetPokemon().targetDamaged().singleSelect()
		.attack(2143, u8"アクアリターン", u8"このポケモンと、ついているすべてのカードを、山札にもどして切る。", "120", { Water, Colorless })
		.textEn(u8"Aqua Return", u8"Shuffle this Pokémon and all attached cards into your deck.")
		.postEffectMe(ToDeckWithAttach)
		.effectShuffle();

	CreateCard(1619, u8"ナミイルカ", Pokemon, 1619)
		.nameEn(u8"Finizen")
		.pokemon(Normal, Basic, Water, 80, 2)
		.weakness(Lightning)
		.attack(2144, u8"ドレインフィン", u8"このポケモンのHPを「20」回復する。", "20", { Water, Water })
		.textEn(u8"Draining Fin", u8"Heal 20 damage from this Pokémon.")
		.postEffectMe(Heal).eVal(20);

	CreateCard(1620, u8"イルカマン", Pokemon, 1620)
		.nameEn(u8"Palafin")
		.pokemon(Normal, Stage1, Water, 150, 2)
		.evolvesFrom(u8"ナミイルカ")
		.weakness(Lightning)
		.attack(2145, u8"ジャスティスナックル", u8"相手のサイドの残り枚数が1枚なら、200ダメージ追加。", "80+", { Water, Water })
		.textEn(u8"Knuckle Justice", u8"If your opponent has exactly 1 Prize card remaining, this attack does 200 more damage.")
		.setPreEffect()
		.condition(ConditionType::CountTarget, 1).targetPlayer(Enemy).targetPrize()
		.preEffectAttackDamageChange(200);

	CreateCard(1621, u8"ラクライ", Pokemon, 1621)
		.nameEn(u8"Electrike")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2146, u8"もってくる", u8"自分の山札を1枚引く。", "", { Lightning })
		.textEn(u8"Collect", u8"Draw a card.")
		.setPostEffect()
		.effectDraw(1)
		.attack(2147, u8"たいあたり", u8"", "30", { Lightning, Lightning })
		.textEn(u8"Tackle", u8"");

	CreateCard(1622, u8"デンヂムシ", Pokemon, 1622)
		.nameEn(u8"Charjabug")
		.pokemon(Normal, Stage1, Lightning, 100, 2)
		.evolvesFrom(u8"アゴジムシ")
		.weakness(Fighting)
		.attack(2148, u8"はさむ", u8"", "30", { Lightning })
		.textEn(u8"Vise Grip", u8"")
		.attack(2149, u8"ぶつかる", u8"", "50", { Lightning, Lightning })
		.textEn(u8"Ram", u8"");

	CreateCard(1623, u8"クワガノン", Pokemon, 1623)
		.nameEn(u8"Vikavolt")
		.pokemon(Normal, Stage2, Lightning, 160, 2)
		.evolvesFrom(u8"デンヂムシ")
		.weakness(Fighting)
		.attack(2150, u8"クイックダイブ", u8"相手のポケモン1匹に、50ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Lightning })
		.textEn(u8"Quick Dive", u8"This attack does 50 damage to 1 of your opponent’s Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffectDamagePokemon(50)
		.attack(2151, u8"ギガレールガン", u8"このポケモンに「ボルト【雷】エネルギー」がついていないなら、このワザは失敗。", "260", { Lightning, Lightning })
		.textEn(u8"Giga Railgun", u8"If this Pokémon has no Voltaic {L} Energy attached, this attack does nothing.")
		.preEffectFailAttack()
		.exist(AreaType::Me).targetNameCondition(TargetType::IsAttachedEnergyName, u8"ボルト雷エネルギー");

	CreateCard(1624, u8"メガゼラオラex", Pokemon, 1624)
		.nameEn(u8"Mega Zeraora ex")
		.pokemon(MegaEx, Basic, Lightning, 270, 1)
		.weakness(Fighting)
		.attack(2152, u8"サンダーフィスト", u8"このポケモンについている【雷】エネルギーの数×60ダメージ。", "60×", { Lightning })
		.textEn(u8"Thunderous Fist", u8"This attack does 60 damage for each {L} Energy attached to this Pokémon.")
		.preEffectMe(AttackDamageChangeTypeEnergyCount).eVal((int)EnergyType::Lightning, 60)
		.attack(2153, u8"ゼプトターン", u8"このポケモンをベンチポケモンと入れ替える。", "150", { Lightning, Lightning, Lightning })
		.textEn(u8"Zepto Turn", u8"Switch this Pokémon with 1 of your Benched Pokémon.")
		.setPostEffect()
		.effectSwitch(Me);

	// Photon Cord is Heavy Baton's on-Knock-Out energy move (SelectSwitchEnergyCard on
	// the Knocked Out Pokémon's Basic {L} Energy), sent to a single Benched Pokémon
	// with SwitchSelectedCard as Yanmega ex's Jet Cyclone does.
	CreateCard(1625, u8"ミライドン", Pokemon, 1625)
		.nameEn(u8"Miraidon")
		.pokemon(Normal, Basic, Lightning, 120, 1)
		.weakness(Fighting)
		.abilityActive(542, u8"フォトンコード", u8"このポケモンが、バトル場で相手のポケモンからワザのダメージを受けてきぜつしたとき、このポケモンについている「基本【雷】エネルギー」を2枚まで選び、ベンチポケモン1匹につけ替える。")
		.textEn(u8" Photon Cord", u8"If this Pokémon is in the Active Spot and is Knocked Out by damage from an attack from your opponent’s Pokémon, move up to 2 Basic {L} Energy cards from this Pokémon to 1 of your Benched Pokémon.")
		.triggerMe(TriggerType::KoEnemyAttackDamageActive)
		.exist(AreaType::Energy, Me).targetCondition(TargetType::AttachedTriggerSubject).targetCardId(LIGHTNING_ENERGY)
		.existMyBench().targetCondition(TargetType::TriggerSubject, 0, NotEqual)
		.effect(SelectSwitchEnergyCard, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedTriggerSubject).targetCardId(LIGHTNING_ENERGY).maxSelectEnergy(2)
		.effect(SwitchSelectedCard, Me).targetBench().targetCondition(TargetType::TriggerSubject, 0, NotEqual).singleSelect()
		.attack(2154, u8"かみなり", u8"このポケモンにも30ダメージ。", "90", { Lightning, Lightning })
		.textEn(u8"Thunder", u8"This Pokémon also does 30 damage to itself.")
		.postEffectDamageMe(30);

	CreateCard(1626, u8"ヤドン", Pokemon, 1626)
		.nameEn(u8"Slowpoke")
		.pokemon(Normal, Basic, Psychic, 70, 2)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2155, u8"すてほうだい", u8"自分の手札を好きなだけ選び、トラッシュする。", "", { Psychic })
		.textEn(u8"All-You-Can-Yeet", u8"You may discard any number of cards from your hand.")
		.postEffect(ToTrash, Me).targetHand().selectAny()
		.attack(2156, u8"ずつき", u8"", "20", { Colorless, Colorless })
		.textEn(u8"Headbutt", u8"");

	CreateCard(1627, u8"ヤドラン", Pokemon, 1627)
		.nameEn(u8"Slowbro")
		.pokemon(Normal, Stage1, Psychic, 130, 3)
		.evolvesFrom(u8"ヤドン")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2157, u8"スッカラカン", u8"自分の手札が1枚もないなら、160ダメージ追加。", "50+", { Psychic })
		.textEn(u8"All Out", u8"If you have no cards in your hand, this attack does 160 more damage.")
		.setPreEffect()
		.conditionLessEqual(0).targetHand()
		.preEffectAttackDamageChange(160)
		.attack(2158, u8"しねんのずつき", u8"", "110", { Colorless, Colorless, Colorless })
		.textEn(u8"Zen Headbutt", u8"");

	// Shellnado Spin is Bouffalant's Ready to Ram delay skill with 12 counters.
	CreateCard(1628, u8"メガヤドランex", Pokemon, 1628)
		.nameEn(u8"Mega Slowbro ex")
		.pokemon(MegaEx, Stage1, Psychic, 330, 3)
		.evolvesFrom(u8"ヤドン")
		.weakness(Darkness)
		.resistance(Fighting)
		.delaySkill(543, u8"シェルネードスピン", u8"次の相手の番、このポケモンがワザのダメージを受けたとき、ワザを使ったポケモンにダメカンを12個のせる。")
		.textEn(u8"Shellnado Spin", u8"During your opponent’s next turn, if this Pokémon is damaged by an attack (even if this Pokémon is Knocked Out), place 12 damage counters on the Attacking Pokémon.")
		.triggerMe(TriggerType::DamagedEnemyAttackActive)
		.effectTriggerObject(DamageCounter).eVal(12)
		.attack(2159, u8"シェルネードスピン", u8"次の相手の番、このポケモンがワザのダメージを受けたとき、ワザを使ったポケモンにダメカンを12個のせる。", "180", { Psychic, Psychic, Psychic })
		.textEn(u8"Shellnado Spin", u8"During your opponent’s next turn, if this Pokémon is damaged by an attack (even if this Pokémon is Knocked Out), place 12 damage counters on the Attacking Pokémon.")
		.postEffectMe(DelayEffect);

	// Intense Kiss is Team Rocket's Grimer's Corrosive Sludge delay skill.
	CreateCard(1629, u8"ルージュラ", Pokemon, 1629)
		.nameEn(u8"Jynx")
		.pokemon(Normal, Basic, Psychic, 100, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.delaySkill(544, u8"きょうれつキッス", u8"次の相手の番の終わりに、このワザを受けたポケモンと、ついているすべてのカードを、トラッシュする。")
		.textEn(u8"Intense Kiss", u8"At the end of your opponent’s next turn, discard the Defending Pokémon and all attached cards.")
		.trigger(TriggerType::TurnEnd)
		.effectTriggerSubject(ToTrash)
		.attack(2160, u8"きょうれつキッス", u8"次の相手の番の終わりに、このワザを受けたポケモンと、ついているすべてのカードを、トラッシュする。", "", { Psychic })
		.textEn(u8"Intense Kiss", u8"At the end of your opponent’s next turn, discard the Defending Pokémon and all attached cards.")
		.postEffect(DelayEffect, Enemy).targetActive()
		.attack(2161, u8"ねんりき", u8"コインを1回投げオモテなら、相手のバトルポケモンを【マヒ】にする。", "50", { Psychic, Colorless })
		.textEn(u8"Psy Bolt", u8"Flip a coin. If heads, your opponent’s Active Pokémon is now Paralyzed.")
		.postEffectParalyzeIfCoinHead();

	CreateCard(1630, u8"マーシャドー", Pokemon, 1630)
		.nameEn(u8"Marshadow")
		.pokemon(Normal, Basic, Psychic, 90, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2162, u8"かげむすび", u8"相手のバトルポケモンのにげるためのエネルギーの数×30ダメージ。", "30×", { Psychic })
		.textEn(u8"Shadowy Knot", u8"This attack does 30 damage for each {C} in your opponent’s Active Pokémon’s Retreat Cost.")
		.preEffect(AttackDamageChangeRetreatCost, Enemy).eVal(30).targetActive();

	CreateCard(1631, u8"コノヨザル", Pokemon, 1631)
		.nameEn(u8"Annihilape")
		.pokemon(Normal, Stage2, Psychic, 150, 2)
		.evolvesFrom(u8"オコリザル")
		.weakness(Darkness)
		.resistance(Fighting)
		.abilityBattleField(545, u8"くちないからだ", u8"このポケモンが、ワザのダメージを受けてきぜつするとき、自分はコインを1回投げる。オモテなら、このポケモンはきぜつせず、残りHPが「10」の状態で場に残る。")
		.textEn(u8" Durable Body", u8"If this Pokémon would be Knocked Out by damage from an attack, flip a coin. If heads, this Pokémon is not Knocked Out, and its remaining HP becomes 10.")
		.triggerMe(TriggerType::PreKo)
		.exist(AreaType::Me).targetHpLessEqual(0)
		.effectBreakIfCoinTail()
		.effectMe(ResetHp).eVal(10)
		.attack(2163, u8"ゴーストブロー", u8"相手のベンチポケモン1匹に、ダメカンを5個のせる。", "100", { Psychic, Psychic })
		.textEn(u8"Ghostly Blow", u8"Place 5 damage counters on 1 of your opponent’s Benched Pokémon.")
		.setPostEffect()
		.existEnemyBench()
		.effect(DamageCounter, Enemy).eVal(5).targetBench().singleSelect();

	CreateCard(1632, u8"マンキー", Pokemon, 1632)
		.nameEn(u8"Mankey")
		.pokemon(Normal, Basic, Fighting, 50, 1)
		.weakness(Psychic)
		.attack(2164, u8"けたぐり", u8"", "20", { Colorless })
		.textEn(u8"Low Kick", u8"");

	CreateCard(1633, u8"オコリザル", Pokemon, 1633)
		.nameEn(u8"Primeape")
		.pokemon(Normal, Stage1, Fighting, 110, 2)
		.evolvesFrom(u8"マンキー")
		.weakness(Psychic)
		.attack(2165, u8"どつく", u8"", "50", { Colorless, Colorless })
		.textEn(u8"Corkscrew Punch", u8"");

	CreateCard(1634, u8"コライドン", Pokemon, 1634)
		.nameEn(u8"Koraidon")
		.pokemon(Normal, Basic, Fighting, 130, 2)
		.weakness(Psychic)
		.attack(2166, u8"バトルクロー", u8"相手のバトルポケモンが進化ポケモンなら、30ダメージ追加。", "30+", { Fighting })
		.textEn(u8"Battle Claw", u8"If your opponent’s Active Pokémon is an Evolution Pokémon, this attack does 30 more damage.")
		.setPreEffect()
		.exist(AreaType::Active, Enemy).targetEvolvedPokemon()
		.preEffectAttackDamageChange(30)
		.attack(2167, u8"ガイアインパクト", u8"このポケモンについているエネルギーを、すべてトラッシュする。", "190", { Fighting, Fighting, Colorless })
		.textEn(u8"Gaia Impact", u8"Discard all Energy from this Pokémon.")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe);

	CreateCard(1635, u8"メガダークライex", Pokemon, 1635)
		.nameEn(u8"Mega Darkrai ex")
		.pokemon(MegaEx, Basic, Darkness, 280, 2)
		.weakness(Grass)
		.attack(2168, u8"ナイトレイド", u8"自分のベンチポケモンにダメカンがのっているなら、110ダメージ追加。", "110+", { Darkness, Darkness })
		.textEn(u8"Dusk Raid", u8"If your Benched Pokémon have any damage counters on them, this attack does 110 more damage.")
		.setPreEffect()
		.existMyBench().targetDamaged()
		.preEffectAttackDamageChange(110)
		.attack(2169, u8"アビスアイ", u8"相手のバトルポケモンが特殊状態なら、そのポケモンをきぜつさせる。", "", { Darkness, Darkness, Darkness })
		.textEn(u8"Abyss Eye", u8"If your opponent’s Active Pokémon is affected by a Special Condition, it is Knocked Out.")
		.setPostEffect()
		.exist(AreaType::Active, Enemy).targetCondition(TargetType::SpecialCondition)
		.effect(Ko, Enemy).targetActive();

	CreateCard(1636, u8"バルチャイ", Pokemon, 1636)
		.nameEn(u8"Vullaby")
		.pokemon(Normal, Basic, Darkness, 70, 1)
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2170, u8"はばたく", u8"", "10", { Darkness })
		.textEn(u8"Flap", u8"")
		.attack(2171, u8"かぜおこし", u8"", "20", { Darkness, Colorless })
		.textEn(u8"Gust", u8"");

	CreateCard(1637, u8"バルジーナ", Pokemon, 1637)
		.nameEn(u8"Mandibuzz")
		.pokemon(Normal, Stage1, Darkness, 120, 2)
		.evolvesFrom(u8"バルチャイ")
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2172, u8"ボーンスナイプ", u8"特殊エネルギーがついている相手のポケモン1匹に、70ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Darkness })
		.textEn(u8"Bone Sniper", u8"This attack does 70 damage to 1 of your opponent’s Pokémon that has any Special Energy attached. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.setPostEffect()
		.existPokemon(Enemy).targetCondition(TargetType::IsAttachedSpecialEnergy)
		.effect(AttackDamage, Enemy).eVal(70).targetPokemon().targetCondition(TargetType::IsAttachedSpecialEnergy).singleSelect()
		.attack(2173, u8"ブラストウインド", u8"", "120", { Darkness, Darkness, Colorless })
		.textEn(u8"Blasting Wind", u8"");

	CreateCard(1638, u8"マーイーカ", Pokemon, 1638)
		.nameEn(u8"Inkay")
		.pokemon(Normal, Basic, Darkness, 60, 1)
		.weakness(Grass)
		.attack(2174, u8"ちょうたつ", u8"自分の山札からグッズを1枚選び、相手に見せて、手札に加える。そして山札を切る。", "", { Darkness })
		.textEn(u8"Procurement", u8"Search your deck for an Item card, reveal it, and put it into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToHandAndShuffle(1).targetItem()
		.attack(2175, u8"かいてんアタック", u8"", "30", { Darkness, Darkness })
		.textEn(u8"Spinning Attack", u8"");

	CreateCard(1639, u8"カラマネロ", Pokemon, 1639)
		.nameEn(u8"Malamar")
		.pokemon(Normal, Stage1, Darkness, 120, 2)
		.evolvesFrom(u8"マーイーカ")
		.weakness(Grass)
		.attack(2176, u8"まどわす", u8"相手のバトルポケモンを【こんらん】にする。", "", { Darkness })
		.textEn(u8"Perplex", u8"Your opponent’s Active Pokémon is now Confused.")
		.postEffect(Confuse, Enemy)
		.attack(2177, u8"ブレインクラッシュ", u8"相手のバトルポケモンが【こんらん】でないなら、このワザは失敗。", "130", { Darkness })
		.textEn(u8"Brain Crush", u8"If your opponent’s Active Pokémon isn’t Confused, this attack does nothing.")
		.preEffectFailAttack()
		.exist(AreaType::Active, Enemy).targetCondition(TargetType::Confuse);

	CreateCard(1640, u8"クスネ", Pokemon, 1640)
		.nameEn(u8"Nickit")
		.pokemon(Normal, Basic, Darkness, 70, 1)
		.weakness(Grass)
		.attack(2178, u8"かじる", u8"", "10", { Darkness })
		.textEn(u8"Gnaw", u8"")
		.attack(2179, u8"うしろげり", u8"", "30", { Darkness, Colorless })
		.textEn(u8"Rear Kick", u8"");

	CreateCard(1641, u8"モルペコex", Pokemon, 1641)
		.nameEn(u8"Morpeko ex")
		.pokemon(Ex, Basic, Darkness, 180, 1)
		.weakness(Grass)
		.attack(2180, u8"ホイールドロー", u8"自分の手札をすべて山札にもどして切る。その後、山札を6枚引く。", "", { Darkness })
		.textEn(u8"Wheely Draw", u8"Shuffle your hand into your deck. Then, draw 6 cards.")
		.setPostEffect()
		.canHandToDeckAndShuffle()
		.effect(ToDeckReverseAndShuffle, Me).targetHand()
		.separator()
		.effectDraw(6)
		.attack(2181, u8"はらぺこボンバー", u8"このポケモンにのっているダメカンの数×40ダメージ追加。", "40+", { Darkness, Darkness })
		.textEn(u8"Hangry Blaster", u8"This attack does 40 more damage for each damage counter on this Pokémon.")
		.preEffectMe(AttackDamageChangeDamageCounter).eVal(40);

	CreateCard(1642, u8"ザルード", Pokemon, 1642)
		.nameEn(u8"Zarude")
		.pokemon(Normal, Basic, Darkness, 130, 2)
		.weakness(Grass)
		.attack(2182, u8"うしろなげ", u8"自分のベンチポケモン1匹にも、30ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "30", { Darkness })
		.textEn(u8"Overhead Throw", u8"This attack also does 30 damage to 1 of your Benched Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffect(AttackDamage, Me).eVal(30).targetBench().singleSelect()
		.attack(2183, u8"シャドーウィップ", u8"自分のベンチポケモンに「シャドー【悪】エネルギー」がついているなら、70ダメージ追加。", "100+", { Darkness, Darkness, Darkness })
		.textEn(u8"Shadowy Whip", u8"If your Benched Pokémon have any Shadowy {D} Energy attached, this attack does 70 more damage.")
		.setPreEffect()
		.existMyBench().targetNameCondition(TargetType::IsAttachedEnergyName, u8"シャドー悪エネルギー")
		.preEffectAttackDamageChange(70);

	CreateCard(1643, u8"オラチフ", Pokemon, 1643)
		.nameEn(u8"Maschiff")
		.pokemon(Normal, Basic, Darkness, 70, 2)
		.weakness(Grass)
		.attack(2184, u8"かみつく", u8"", "40", { Darkness, Darkness })
		.textEn(u8"Bite", u8"");

	CreateCard(1644, u8"マフィティフ", Pokemon, 1644)
		.nameEn(u8"Mabosstiff")
		.pokemon(Normal, Stage1, Darkness, 140, 3)
		.evolvesFrom(u8"オラチフ")
		.weakness(Grass)
		.attack(2185, u8"かみつく", u8"", "60", { Darkness, Darkness })
		.textEn(u8"Bite", u8"")
		.attack(2186, u8"とびこみヘッド", u8"次の相手の番、このポケモンが受けるワザのダメージは「+100」される。", "210", { Darkness, Darkness, Darkness })
		.textEn(u8"Plunging Headbutt", u8"During your opponent’s next turn, this Pokémon takes 100 more damage from attacks (after applying Weakness and Resistance).")
		.postEffectMe(TakeDamageChangeNextEnemyTurn).eVal(100);

	CreateCard(1645, u8"イーユイ", Pokemon, 1645)
		.nameEn(u8"Chi-Yu")
		.pokemon(Normal, Basic, Darkness, 90, 1)
		.weakness(Grass)
		.attack(2187, u8"うずまくねたみ", u8"このポケモンにダメカンが2個以上のっているなら、90ダメージ追加。このワザのダメージは弱点を計算しない。", "20+", { Darkness })
		.textEn(u8"Whirling Envy", u8"If this Pokémon has 2 or more damage counters on it, this attack does 90 more damage. This attack’s damage isn’t affected by Weakness.")
		.noTargetWeaknessOnly()
		.setPreEffect()
		.exist(AreaType::Me).targetCondition(TargetType::DamageCounter, 2, GreaterEqual)
		.preEffectAttackDamageChange(90);

	CreateCard(1646, u8"エアームド", Pokemon, 1646)
		.nameEn(u8"Skarmory")
		.pokemon(Normal, Basic, Metal, 120, 1)
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2188, u8"スチールカッター", u8"自分の手札から「基本【鋼】エネルギー」を2枚までトラッシュし、その枚数×40ダメージ。", "40×", { Metal })
		.textEn(u8"Steel Cutter", u8"Discard up to 2 Basic {M} Energy cards from your hand, and this attack does 40 damage for each card you discarded in this way.")
		.preEffect(ToTrash, Me).targetHand().targetCardId(METAL_ENERGY).maxSelect(2)
		.effectEffectedCard(AttackDamageChangeTargetCount).eVal(40);

	CreateCard(1647, u8"ツツケラ", Pokemon, 1647)
		.nameEn(u8"Pikipek")
		.pokemon(Normal, Basic, Colorless, 70, 1)
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2189, u8"にどづき", u8"コインを2回投げ、オモテの数×10ダメージ。", "10×", { Colorless })
		.textEn(u8"Double Stab", u8"Flip 2 coins. This attack does 10 damage for each heads.")
		.preEffect(AttackDamageChangeCoin, None).eVal(2, 10);

	CreateCard(1648, u8"ケララッパ", Pokemon, 1648)
		.nameEn(u8"Trumbeak")
		.pokemon(Normal, Stage1, Colorless, 90, 1)
		.evolvesFrom(u8"ツツケラ")
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2190, u8"そらをとぶ", u8"コインを1回投げウラなら、このワザは失敗。オモテなら、次の相手の番、このポケモンはワザのダメージや効果を受けない。", "30", { Colorless })
		.textEn(u8"Fly", u8"Flip a coin. If tails, this attack does nothing. If heads, during your opponent’s next turn, prevent all damage from and effects of attacks done to this Pokémon.")
		.preEffectFailAttack()
		.effectBreakIfCoinTail()
		.effectMe(NoDamageAndEffectAttackNextEnemyTurn);

	CreateCard(1649, u8"ドデカバシ", Pokemon, 1649)
		.nameEn(u8"Toucannon")
		.pokemon(Normal, Stage2, Colorless, 150, 2)
		.evolvesFrom(u8"ケララッパ")
		.weakness(Lightning)
		.resistance(Fighting)
		.activateSkillOnceTurn(546, u8"スカイドロー", u8"自分の番に1回使える。自分の山札を1枚引く。")
		.textEn(u8" Aerial Draw", u8"Once during your turn, you may use this Ability. Draw a card.")
		.effectDraw(1)
		.attack(2191, u8"フェザーロンド", u8"おたがいのベンチポケモンの数×20ダメージ追加。", "60+", { Colorless })
		.textEn(u8"Feather Rondo", u8"This attack does 20 more damage for each Benched Pokémon (both yours and your opponent’s).")
		.preEffect(AttackDamageChangeTargetCount, Both).eVal(20).targetBench();

	CreateCard(1650, u8"タイプ：ヌル", Pokemon, 1650)
		.nameEn(u8"Type: Null")
		.pokemon(Normal, Basic, Colorless, 110, 2)
		.weakness(Fighting)
		.attack(2192, u8"パワーエッジ", u8"", "40", { Colorless, Colorless })
		.textEn(u8"Power Edge", u8"");

	CreateCard(1651, u8"シルヴァディ", Pokemon, 1651)
		.nameEn(u8"Silvally")
		.pokemon(Normal, Stage1, Colorless, 140, 2)
		.evolvesFrom(u8"タイプ：ヌル")
		.weakness(Fighting)
		.activateSkillOnceTurn(547, u8"バディコール", u8"自分の手札が1枚もないなら、自分の番に1回使える。自分の山札からサポートを1枚選び、相手に見せて、手札に加える。そして山札を切る。")
		.textEn(u8" Call a Buddy", u8"Once during your turn, if you have no cards in your hand, you may use this Ability. Search your deck for a Supporter card, reveal it, and put it into your hand. Then, shuffle your deck.")
		.conditionLessEqual(0).targetHand()
		.effectDeckToHandAndShuffle(1).targetSupporter()
		.attack(2193, u8"エアスラッシュ", u8"このポケモンについているエネルギーを1個選び、トラッシュする。", "130", { Colorless, Colorless, Colorless })
		.textEn(u8"Air Slash", u8"Discard an Energy from this Pokémon.")
		.postEffectTrashEnergyMe(1);

	CreateCard(1652, u8"オトシドリ", Pokemon, 1652)
		.nameEn(u8"Bombirdier")
		.pokemon(Normal, Basic, Colorless, 100, 1)
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2194, u8"おとどけチャレンジ", u8"コインを2回投げ、すべてオモテなら、自分の山札からポケモンを1枚選び、ベンチに出す。そして山札を切る。", "", { Colorless, Colorless })
		.textEn(u8"Challenging Delivery", u8"Flip 2 coins. If both of them are heads, search your deck for a Pokémon and put it onto your Bench. Then, shuffle your deck.")
		.setPostEffect()
		.effectBreakIfCoinTailMulti(2)
		.effectDeckToBenchAndShuffle(1).targetPokemonCard()
		.attack(2195, u8"スピードウイング", u8"", "100", { Colorless, Colorless, Colorless })
		.textEn(u8"Speed Wing", u8"");

	CreateCard(1653, u8"ダークベル", Item, 1653)
		.nameEn(u8"Dark Bell")
		.playSkill(548, u8"おたがいのバトルポケモン（【悪】ポケモンをのぞく）を、それぞれ【こんらん】にする。")
		.textEn(u8"Dark Bell", u8"Both Active non-{D} Pokémon are now Confused.")
		.existAnyTargetAfterEffect()
		.effect(Confuse, Both).targetActive().targetCondition(TargetType::EnergyType, (int)EnergyType::Darkness, NotEqual);

	// The +80 is put on each of your non-Rule-Box Pokémon in play when the card is
	// played (DamageChangeThisTurn, which CalcDamage applies only against the
	// opponent's Active). A Pokémon put into play later in the same turn does not
	// get it -- the one place this differs from the printed text.
	CreateCard(1654, u8"グラジオの決戦", Supporter, 1654)
		.nameEn(u8"Gladion’s Final Battle")
		.playSkill(549, u8"このカードは、自分の手札がこのカード1枚だけのときにしか使えない。この番、自分のポケモン（「ルールを持つポケモン」をのぞく）が使うワザの、相手のバトルポケモンへのダメージは「+80」される。")
		.textEn(u8"Gladion’s Final Battle", u8"You can use this card only when it is the last card in your hand.\n\nDuring this turn, attacks used by your Pokémon that don’t have a Rule Box do 80 more damage to your opponent’s Active Pokémon (before applying Weakness and Resistance). (Pokémon ex, Pokémon V, etc. have Rule Boxes.)")
		.conditionLessEqual(0).targetHand().notMe()
		.effect(DamageChangeThisTurn, Me).eVal(80).targetPokemon().targetNotRulePokemon();

	CreateCard(1655, u8"ジェット", Supporter, 1655)
		.nameEn(u8"Jett")
		.playSkill(550, u8"相手の場の「メガシンカex」の数ぶん、自分の山札を引く。")
		.textEn(u8"Jett", u8"Draw a card for each of your opponent’s Mega Evolution Pokémon ex in play.")
		.existMyDeck()
		.existPokemon(Enemy).targetCondition(TargetType::MegaEx)
		.effect(NoEffect, Enemy).targetPokemon().targetCondition(TargetType::MegaEx)
		.effectDraw(1).multiplyEffectValuePreTargetCount();

	CreateCard(1656, u8"カスミの元気", Supporter, 1656)
		.nameEn(u8"Misty’s Vitality")
		.playSkill(551, u8"このカードを使ったなら、自分の番は終わる。自分の山札から「基本【水】エネルギー」を4枚まで選び、自分のポケモン1匹につける。そして山札を切る。")
		.textEn(u8"Misty’s Vitality", u8"Search your deck for up to 4 Basic {W} Energy cards and attach them to 1 of your Pokémon. Then, shuffle your deck. Your turn ends.")
		.effectSelectAttachEnergyDeck(4).targetCardId(WATER_ENERGY)
		.effect(AttachSelectedCard, Me).targetPokemon().singleSelect().seeingDeck()
		.effectShuffle()
		.effectTurnEnd();

	CreateCard(1657, u8"サビ組のしたっぱ", Supporter, 1657)
		.nameEn(u8"Rust Syndicate Grunt")
		.playSkill(552, u8"このカードは、前の相手の番に、自分のポケモンがきぜつしていなければ使えない。相手の場のポケモンについているエネルギーを1個選び、トラッシュする。")
		.textEn(u8"Rust Syndicate Grunt", u8"You can use this card only if any of your Pokémon were Knocked Out during your opponent’s last turn.\n\nDiscard an Energy from 1 of your opponent’s Pokémon.")
		.condition(ConditionType::KoPreEnemyTurn)
		.exist(AreaType::Energy, Enemy)
		.effect(ToTrash, Enemy).targetAttachedEnergy().selectEnergy(1);

	CreateCard(1658, u8"シャドー悪エネルギー", SpecialEnergy, 1658)
		.nameEn(u8"Shadowy Darkness Energy")
		.energySkill(553, u8"このカードは、ポケモンについているかぎり、【悪】エネルギー1個ぶんとしてはたらく。このカードをつけている【悪】ポケモンは、ベンチにいるかぎり、相手のワザのダメージを受けない。")
		.textEn(u8"Shadowy Darkness Energy", u8"As long as this card is attached to a Pokémon, it provides {D} Energy.\n\nAs long as the {D} Pokémon this card is attached to is on your Bench, prevent all damage done to it by attacks from your opponent’s Pokémon.")
		.effectEnergyContinual(NoDamageEnemyAttack).targetEnergyType(Darkness).targetCondition(TargetType::Area, (int)AreaType::Bench)
		.specialEnergy(EnergyType::Darkness, 1);

	CreateCard(1659, u8"ボルト雷エネルギー", SpecialEnergy, 1659)
		.nameEn(u8"Voltaic Lightning Energy")
		.energySkill(554, u8"このカードは、ポケモンについているかぎり、【雷】エネルギー1個ぶんとしてはたらく。このカードをつけている【雷】ポケモンが使うワザの、相手のバトルポケモンへのダメージは「+20」される。")
		.textEn(u8"Voltaic Lightning Energy", u8"As long as this card is attached to a Pokémon, it provides {L} Energy.\n\nAttacks used by the {L} Pokémon this card is attached to do 20 more damage to your opponent’s Active Pokémon (before applying Weakness and Resistance).")
		.effectEnergyContinual(DamageChangeActive).eVal(20).targetEnergyType(Lightning)
		.specialEnergy(EnergyType::Lightning, 1);

	// ---- 30th Celebration (30C) ----
	CreateCard(1660, u8"タマタマ", Pokemon, 1660)
		.nameEn(u8"Exeggcute")
		.pokemon(Normal, Basic, Grass, 60, 1)
		.weakness(Fire)
		.attack(2196, u8"さいみんじゅつ", u8"相手のバトルポケモンを【ねむり】にする。", "", { Colorless })
		.textEn(u8"Hypnosis", u8"Your opponent’s Active Pokémon is now Asleep.")
		.postEffect(Sleep, Enemy);

	CreateCard(1661, u8"バルビート", Pokemon, 1661)
		.nameEn(u8"Volbeat")
		.pokemon(Normal, Basic, Grass, 80, 1)
		.weakness(Fire)
		.attack(2197, u8"さそうひかり", u8"相手のベンチポケモンを1匹選び、バトルポケモンと入れ替える。", "", { Grass })
		.textEn(u8"Luring Glow", u8"Switch in 1 of your opponent’s Benched Pokémon to the Active Spot.")
		.setPostEffect()
		.effectSwitchEnemyBench()
		.attack(2198, u8"むしのさざめき", u8"", "90", { Colorless, Colorless, Colorless })
		.textEn(u8"Bug Buzz", u8"");

	CreateCard(1662, u8"トロピウス", Pokemon, 1662)
		.nameEn(u8"Tropius")
		.pokemon(Normal, Basic, Grass, 120, 2)
		.weakness(Fire)
		.attack(2199, u8"まきかえす", u8"前の相手の番に、ワザのダメージで、自分のポケモンがきぜつしていたなら、90ダメージ追加。", "30+", { Grass, Colorless })
		.textEn(u8"Rally Back", u8"If any of your Pokémon were Knocked Out by damage from an attack during your opponent’s last turn, this attack does 90 more damage.")
		.setPreEffect()
		.condition(ConditionType::KoAttackDamagePreEnemyTurn)
		.preEffectAttackDamageChange(90)
		.attack(2200, u8"カッターウインド", u8"", "90", { Grass, Colorless, Colorless })
		.textEn(u8"Cutting Wind", u8"");

	CreateCard(1663, u8"チェリンボ", Pokemon, 1663)
		.nameEn(u8"Cherubi")
		.pokemon(Normal, Basic, Grass, 40, 1)
		.weakness(Fire)
		.attack(2201, u8"かくれる", u8"コインを1回投げオモテなら、次の相手の番、このポケモンはワザのダメージや効果を受けない。", "", { Colorless })
		.textEn(u8"Hide", u8"Flip a coin. If heads, during your opponent’s next turn, prevent all damage from and effects of attacks done to this Pokémon.")
		.setPostEffect()
		.effectBreakIfCoinTail()
		.effectMe(NoDamageAndEffectAttackNextEnemyTurn)
		.attack(2202, u8"はねまわる", u8"", "10", { Grass })
		.textEn(u8"Flop", u8"");

	CreateCard(1664, u8"チェリム", Pokemon, 1664)
		.nameEn(u8"Cherrim")
		.pokemon(Normal, Stage1, Grass, 80, 1)
		.evolvesFrom(u8"チェリンボ")
		.weakness(Fire)
		.attack(2203, u8"エナジーギフト", u8"自分の山札から基本エネルギーを2枚まで選び、自分のポケモンに好きなようにつける。そして山札を切る。", "", { Colorless })
		.textEn(u8"Energy Gift", u8"Search your deck for up to 2 Basic Energy cards and attach them to your Pokémon in any way you like. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckAttachEnergyAndShuffle(2).targetBasicEnergy()
		.attack(2204, u8"このは", u8"", "50", { Grass, Colorless })
		.textEn(u8"Leafage", u8"");

	CreateCard(1665, u8"ビビヨン", Pokemon, 1665)
		.nameEn(u8"Vivillon")
		.pokemon(Normal, Stage2, Grass, 120, 1)
		.evolvesFrom(u8"コフーライ")
		.weakness(Fire)
		.activateSkillOnceTurn(555, u8"みちびきのまい", u8"自分の番に1回使える。コインを1回投げオモテなら、自分の山札からポケモンを1枚選び、相手に見せて、手札に加える。そして山札を切る。")
		.textEn(u8" Guiding Dance", u8"Once during your turn, you may use this Ability. Flip a coin. If heads, search your deck for a Pokémon, reveal it, and put it into your hand. Then, shuffle your deck.")
		.existMyDeck()
		.effectBreakIfCoinTail()
		.effectDeckToHandAndShuffle(1).targetPokemonCard()
		.attack(2205, u8"どくのこな", u8"相手のバトルポケモンを【どく】にする。", "60", { Grass, Colorless })
		.textEn(u8"Poison Powder", u8"Your opponent’s Active Pokémon is now Poisoned.")
		.postEffect(Poison, Enemy);

	CreateCard(1666, u8"ロコン", Pokemon, 1666)
		.nameEn(u8"Vulpix")
		.pokemon(Normal, Basic, Fire, 70, 1)
		.weakness(Water)
		.attack(2206, u8"けりつける", u8"コインを1回投げウラなら、このワザは失敗。", "30", { Fire })
		.textEn(u8"Wild Kick", u8"Flip a coin. If tails, this attack does nothing.")
		.preEffectFailAttackCoinTail();

	CreateCard(1667, u8"キュウコン", Pokemon, 1667)
		.nameEn(u8"Ninetales")
		.pokemon(Normal, Stage1, Fire, 110, 1)
		.evolvesFrom(u8"ロコン")
		.weakness(Water)
		.attack(2207, u8"ほのおのしっぽ", u8"", "60", { Fire })
		.textEn(u8"Flame Tail", u8"");

	CreateCard(1668, u8"ファイヤー", Pokemon, 1668)
		.nameEn(u8"Moltres")
		.pokemon(Normal, Basic, Fire, 120, 1)
		.weakness(Water)
		.activateSkillOnceTurn(556, u8"もえるはばたき", u8"自分の場に「フリーザー」「サンダー」がいるなら、自分の番に1回使える。自分の手札から「基本【炎】エネルギー」を1枚選び、このポケモンにつける。")
		.textEn(u8" Fiery Flapping", u8"Once during your turn, if you have Articuno and Zapdos in play, you may use this Ability. Attach a Basic {R} Energy card from your hand to this Pokémon.")
		.existPokemon().targetName(u8"フリーザー")
		.existPokemon().targetName(u8"サンダー")
		.effectHandAttachEnergyMe().targetCardId(FIRE_ENERGY)
		.attack(2208, u8"ほのおのうず", u8"このポケモンについているエネルギーを2個選び、トラッシュする。", "130", { Fire, Fire, Colorless })
		.textEn(u8"Fire Spin", u8"Discard 2 Energy from this Pokémon.")
		.postEffectTrashEnergyMe(2);

	CreateCard(1669, u8"ホウオウ", Pokemon, 1669)
		.nameEn(u8"Ho-Oh")
		.pokemon(Normal, Basic, Fire, 130, 2)
		.weakness(Water)
		.attack(2209, u8"せいなるいぶき", u8"このポケモンについているエネルギーを、すべてトラッシュする。自分のベンチポケモン1匹のHPを、すべて回復する。", "", { Fire, Fire })
		.textEn(u8"Sacred Breath", u8"Discard all Energy from this Pokémon. Heal all damage from 1 of your Benched Pokémon.")
		.postEffectTrashEnergyMeAll()
		.effect(HealAll, Me).targetBench().targetDamaged().singleSelect()
		.attack(2210, u8"ほのおのつばさ", u8"", "100", { Fire, Fire, Fire })
		.textEn(u8"Fire Wing", u8"");

	CreateCard(1670, u8"ビクティニ", Pokemon, 1670)
		.nameEn(u8"Victini")
		.pokemon(Normal, Basic, Fire, 80, 1)
		.weakness(Water)
		.attack(2211, u8"なかまをよぶ", u8"自分の山札から【たね】ポケモンを2枚まで選び、ベンチに出す。そして山札を切る。", "", { Colorless })
		.textEn(u8"Call for Family", u8"Search your deck for up to 2 Basic Pokémon and put them onto your Bench. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToBenchAndShuffle(2).targetBasicPokemon()
		.attack(2212, u8"Vフレイム", u8"", "50", { Fire, Colorless })
		.textEn(u8"V-Flame", u8"");

	CreateCard(1671, u8"レシラム", Pokemon, 1671)
		.nameEn(u8"Reshiram")
		.pokemon(Normal, Basic, Fire, 130, 2)
		.weakness(Water)
		.attack(2213, u8"きりさく", u8"", "50", { Fire, Colorless })
		.textEn(u8"Slash", u8"")
		.attack(2214, u8"レーザーフレイム", u8"このポケモンに【雷】エネルギーがついているなら、80ダメージ追加。", "80+", { Fire, Colorless, Colorless })
		.textEn(u8"Laser Flame", u8"If this Pokémon has any {L} Energy attached, this attack does 80 more damage.")
		.setPreEffect()
		.conditionAttachEnergyMe(Lightning)
		.preEffectAttackDamageChange(80);

	CreateCard(1672, u8"ホゲータex", Pokemon, 1672)
		.nameEn(u8"Fuecoco ex")
		.pokemon(Ex, Basic, Fire, 210, 2)
		.weakness(Water)
		.attack(2215, u8"こがす", u8"相手のバトルポケモンを【やけど】にする。", "", { Fire })
		.textEn(u8"Singe", u8"Your opponent’s Active Pokémon is now Burned.")
		.postEffect(Burn, Enemy)
		.attack(2216, u8"ごきげんフレイム", u8"自分がすでにとったサイドの枚数×70ダメージ。", "70×", { Fire, Fire, Colorless })
		.textEn(u8"Cheerful Flame", u8"This attack does 70 damage for each Prize card you have taken.")
		.preEffect(AttackDamageChangeTakenPrize, Me).eVal(70);

	CreateCard(1673, u8"ヤドン", Pokemon, 1673)
		.nameEn(u8"Slowpoke")
		.pokemon(Normal, Basic, Water, 80, 2)
		.weakness(Lightning)
		.attack(2217, u8"いどにかくれる", u8"コインを1回投げオモテなら、次の相手の番、このポケモンはワザのダメージや効果を受けない。", "", { Colorless })
		.textEn(u8"Well-Hidden", u8"Flip a coin. If heads, during your opponent’s next turn, prevent all damage from and effects of attacks done to this Pokémon.")
		.setPostEffect()
		.effectBreakIfCoinTail()
		.effectMe(NoDamageAndEffectAttackNextEnemyTurn)
		.attack(2218, u8"みずでっぽう", u8"", "20", { Water, Colorless })
		.textEn(u8"Water Gun", u8"");

	CreateCard(1674, u8"ラプラス", Pokemon, 1674)
		.nameEn(u8"Lapras")
		.pokemon(Normal, Basic, Water, 130, 2)
		.weakness(Metal)
		.attack(2219, u8"のせておよぐ", u8"自分の山札からサポートを1枚選び、相手に見せて、手札に加える。そして山札を切る。", "", { Colorless })
		.textEn(u8"Ferry Across", u8"Search your deck for a Supporter card, reveal it, and put it into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToHandAndShuffle(1).targetSupporter()
		.attack(2220, u8"れいとうビーム", u8"コインを1回投げオモテなら、相手のバトルポケモンを【マヒ】にする。", "80", { Water, Colorless, Colorless })
		.textEn(u8"Ice Beam", u8"Flip a coin. If heads, your opponent’s Active Pokémon is now Paralyzed.")
		.postEffectParalyzeIfCoinHead();

	CreateCard(1675, u8"フリーザー", Pokemon, 1675)
		.nameEn(u8"Articuno")
		.pokemon(Normal, Basic, Water, 120, 1)
		.weakness(Metal)
		.activateSkillOnceTurn(557, u8"いてつくはばたき", u8"自分の場に「ファイヤー」「サンダー」がいるなら、自分の番に1回使える。自分の手札から「基本【水】エネルギー」を1枚選び、このポケモンにつける。")
		.textEn(u8" Frosty Flapping", u8"Once during your turn, if you have Moltres and Zapdos in play, you may use this Ability. Attach a Basic {W} Energy card from your hand to this Pokémon.")
		.existPokemon().targetName(u8"ファイヤー")
		.existPokemon().targetName(u8"サンダー")
		.effectHandAttachEnergyMe().targetCardId(WATER_ENERGY)
		.attack(2221, u8"あられ", u8"相手のポケモン全員に、それぞれ30ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Water, Water, Colorless })
		.textEn(u8"Hail", u8"This attack does 30 damage to each of your opponent’s Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffect(AttackDamage, Enemy).eVal(30).targetPokemon();

	CreateCard(1676, u8"カイオーガ", Pokemon, 1676)
		.nameEn(u8"Kyogre")
		.pokemon(Normal, Basic, Water, 140, 4)
		.weakness(Lightning)
		.attack(2222, u8"ハイドロポンプ", u8"このポケモンについている【水】エネルギーの数×30ダメージ追加。", "60+", { Colorless, Colorless, Colorless, Colorless })
		.textEn(u8"Hydro Pump", u8"This attack does 30 more damage for each {W} Energy attached to this Pokémon.")
		.preEffectMe(AttackDamageChangeTypeEnergyCount).eVal((int)EnergyType::Water, 30);

	CreateCard(1677, u8"パルキア", Pokemon, 1677)
		.nameEn(u8"Palkia")
		.pokemon(Normal, Basic, Water, 130, 2)
		.weakness(Lightning)
		.attack(2223, u8"ワームホール", u8"このポケモンをベンチポケモンと入れ替える。その後、相手は相手自身のバトルポケモンをベンチポケモンと入れ替える。", "100", { Water, Water, Colorless })
		.textEn(u8"Wormhole", u8"Switch this Pokémon with 1 of your Benched Pokémon. If you do, switch out your oppoennt’s Active Pokémon to the Bench. (Your opponent chooses the new Active Pokémon.)")
		.setPostEffect()
		.effectSwitch(Me)
		.separator()
		.effectSwitch(Enemy).enemySelect().effectTargetActive();

	// Stealthy Slash is Team Rocket's Sneasel's Strike the Sleeper, able to pick the
	// Active Pokémon too.
	CreateCard(1678, u8"ゲッコウガex", Pokemon, 1678)
		.nameEn(u8"Greninja ex")
		.pokemon(Ex, Stage2, Water, 300, 1)
		.evolvesFrom(u8"ゲコガシラ")
		.weakness(Lightning)
		.attack(2224, u8"おんみつぎり", u8"相手のポケモン1匹に、そのポケモンにのっているダメカンの数×30ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Water })
		.textEn(u8"Stealthy Slash", u8"This attack does 30 damage to 1 of your opponent’s Pokémon for each damage counter on that Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.setPostEffect()
		.existPokemon(Enemy)
		.postEffect(NoEffect, Enemy).targetPokemon().singleSelect().setContext(SelectContext::Damage)
		.effectEffectedCard(EffectDamageChangeDamageCounter).eVal(30)
		.effectEffectedCard(AttackDamage).eVal(0)
		.attack(2225, u8"アクアエッジ", u8"", "160", { Water, Water })
		.textEn(u8"Aqua Edge", u8"");

	CreateCard(1679, u8"ヨワシ", Pokemon, 1679)
		.nameEn(u8"Wishiwashi")
		.pokemon(Normal, Basic, Water, 30, 1)
		.weakness(Lightning)
		.abilityBattleField(558, u8"むれのはんげき", u8"このポケモンがいるかぎり、自分のバトル場の「ヨワシ（『ポケモンex』をふくむ）」が、相手のポケモンからワザのダメージを受けたとき、ワザを使ったポケモンにダメカンを3個のせる。")
		.textEn(u8" Counterattack Grouping", u8"If your Wishiwashi or Wishiwashi ex is in the Active Spot and is damaged by an attack from your opponent’s Pokémon (even if your Pokémon is Knocked Out), place 3 damage counters on the Attacking Pokémon.")
		.trigger(TriggerType::DamagedEnemyAttackActive).targetActive().targetPlayer(Me).targetNameContains(u8"ヨワシ")
		.effectTriggerObject(DamageCounter).eVal(3)
		.attack(2226, u8"ふいをつく", u8"コインを1回投げウラなら、このワザは失敗。", "30", { Water })
		.textEn(u8"Surprise Attack", u8"Flip a coin. If tails, this attack does nothing.")
		.preEffectFailAttackCoinTail();

	CreateCard(1680, u8"ピカチュウ", Pokemon, 1680)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2227, u8"でんきショック", u8"コインを1回投げオモテなら、相手のバトルポケモンを【マヒ】にする。", "20", { Lightning, Colorless })
		.textEn(u8"Thunder Shock", u8"Flip a coin. If heads, your opponent’s Active Pokémon is now Paralyzed.")
		.postEffectParalyzeIfCoinHead();

	CreateCard(1681, u8"ピカチュウ", Pokemon, 1681)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2228, u8"ボルテッカー", u8"このポケモンにも30ダメージ。", "80", { Lightning, Colorless, Colorless })
		.textEn(u8"Volt Tackle", u8"This Pokémon also does 30 damage to itself.")
		.postEffectDamageMe(30);

	CreateCard(1682, u8"ピカチュウ", Pokemon, 1682)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 2)
		.weakness(Fighting)
		.attack(2229, u8"スパーク", u8"相手のベンチポケモン1匹にも、20ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "20", { Lightning, Colorless })
		.textEn(u8"Spark", u8"This attack also does 20 damage to 1 of your opponent’s Benched Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffectDamageBench(20);

	CreateCard(1683, u8"ピカチュウ", Pokemon, 1683)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 2)
		.weakness(Fighting)
		.attack(2230, u8"のぞきこむ", u8"相手の手札を見る。", "", { Colorless })
		.textEn(u8"Peer At", u8"Your opponent reveals their hand.")
		.setPostEffect()
		.effectLookEnemyhand();

	CreateCard(1684, u8"ピカチュウ", Pokemon, 1684)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 80, 3)
		.weakness(Fighting)
		.attack(2231, u8"ひとやすみ", u8"このポケモンのHPを「30」回復する。", "", { Colorless })
		.textEn(u8"Nap", u8"Heal 30 damage from this Pokémon.")
		.postEffectMe(Heal).eVal(30);

	CreateCard(1685, u8"ピカチュウ", Pokemon, 1685)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.abilityActive(559, u8"さびしいめせん", u8"このポケモンがバトル場にいるかぎり、相手のバトルポケモンが使うワザのダメージは「-20」される。")
		.textEn(u8" Lonely Gaze", u8"As long as this Pokémon is in the Active Spot, attacks used by your opponent’s Active Pokémon do 20 less damage (before applying Weakness and Resistance).")
		.effect(DamageChange, Enemy).eVal(-20).targetActive()
		.attack(2232, u8"ピカボール", u8"", "20", { Lightning, Colorless })
		.textEn(u8"Pika Ball", u8"");

	CreateCard(1686, u8"ピカチュウ", Pokemon, 1686)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2233, u8"ともだちをさがす", u8"自分の山札からポケモンを1枚選び、相手に見せて、手札に加える。そして山札を切る。", "", { Colorless })
		.textEn(u8"Find a Friend", u8"Search your deck for a Pokémon, reveal it, and put it into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToHandAndShuffle(1).targetPokemonCard();

	CreateCard(1687, u8"ピカチュウ", Pokemon, 1687)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2234, u8"マッハボルト", u8"", "30", { Lightning })
		.textEn(u8"Mach Bolt", u8"");

	CreateCard(1688, u8"ピカチュウ", Pokemon, 1688)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 50, 0)
		.weakness(Fighting)
		.attack(2235, u8"かじる", u8"", "10", { Colorless })
		.textEn(u8"Gnaw", u8"");

	CreateCard(1689, u8"ピカチュウ", Pokemon, 1689)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 50, 1)
		.weakness(Fighting)
		.attack(2236, u8"にげまわる", u8"このポケモンをベンチポケモンと入れ替える。", "", { Colorless })
		.textEn(u8"Scurry About", u8"Switch this Pokémon with 1 of your Benched Pokémon.")
		.setPostEffect()
		.effectSwitch(Me);

	CreateCard(1690, u8"ピカチュウ", Pokemon, 1690)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.abilityBench(560, u8"みをかくす", u8"このポケモンは、ベンチにいるかぎり、相手のポケモンからワザのダメージや効果を受けない。")
		.textEn(u8" Keep Hidden", u8"As long as this Pokémon is on your Bench, prevent all damage from and effects of attacks from your opponent’s Pokémon done to this Pokémon.")
		.effectMe(NoDamageAndEffectEnemyAttack)
		.attack(2237, u8"プチでんき", u8"", "10", { Lightning })
		.textEn(u8"Tiny Charge", u8"");

	CreateCard(1691, u8"ピカチュウ", Pokemon, 1691)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 2)
		.weakness(Fighting)
		.attack(2238, u8"ピカれんさ", u8"自分の場の「ピカチュウ（『ポケモンex』をふくむ）」の数×40ダメージ。", "40×", { Lightning, Lightning, Lightning })
		.textEn(u8"Pika Chain", u8"This attack does 40 damage for each of your Pikachu and Pikachu ex in play.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(40).targetPokemon().targetNameContains(u8"ピカチュウ");

	CreateCard(1692, u8"ピカチュウ", Pokemon, 1692)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2239, u8"ころがる", u8"", "30", { Colorless, Colorless })
		.textEn(u8"Rollout", u8"");

	CreateCard(1693, u8"ピカチュウ", Pokemon, 1693)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 2)
		.weakness(Fighting)
		.attack(2240, u8"ちょっとつっこむ", u8"このポケモンにも10ダメージ。", "40", { Lightning, Colorless })
		.textEn(u8"Slight Intrusion", u8"This Pokémon also does 10 damage to itself.")
		.postEffectDamageMe(10);

	CreateCard(1694, u8"ピカチュウ", Pokemon, 1694)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2241, u8"エナジーテール", u8"自分の山札からエネルギーを1枚選び、相手に見せて、手札に加える。そして山札を切る。", "", { Colorless })
		.textEn(u8"Energized Tail", u8"Search your deck for an Energy card, reveal it, and put it into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToHandAndShuffle(1).targetEnergyCard()
		.attack(2242, u8"ピカパンチ", u8"", "30", { Lightning, Colorless })
		.textEn(u8"Pika Punch", u8"");

	CreateCard(1695, u8"ピカチュウ", Pokemon, 1695)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2243, u8"ねらってスパーク", u8"相手のポケモン1匹に、20ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Lightning })
		.textEn(u8"Targeted Spark", u8"This attack does 20 damage to 1 of your opponent’s Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffectDamagePokemon(20);

	CreateCard(1696, u8"ピカチュウ", Pokemon, 1696)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2244, u8"アイアンテール", u8"ウラが出るまでコインを投げ、オモテの数×20ダメージ。", "20×", { Colorless })
		.textEn(u8"Iron Tail", u8"Flip a coin until you get tails. This attack does 20 damage for each heads.")
		.preEffect(AttackDamageChangeCoinUntilTail, None).eVal(20);

	CreateCard(1697, u8"ピカチュウ", Pokemon, 1697)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2245, u8"ぶらさがる", u8"", "10", { Colorless })
		.textEn(u8"Hang Down", u8"")
		.attack(2246, u8"エレキック", u8"", "40", { Lightning, Colorless, Colorless })
		.textEn(u8"Zap Kick", u8"");

	CreateCard(1698, u8"ピカチュウ", Pokemon, 1698)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2247, u8"じゅうでんダッシュ", u8"ウラが出るまでコインを投げ、オモテの数ぶんまで、自分の山札から「基本【雷】エネルギー」を選び、このポケモンにつける。そして山札を切る。", "", { Colorless })
		.textEn(u8"Charge-Up Dash", u8"Flip a coin until you get tails. Search your deck for an amount of Basic {L} Energy up to the number of heads and attach it to this Pokémon. Then, shuffle your deck.")
		.setPostEffect()
		.exist(AreaType::Deck)
		.effect(CoinUntilTail, None).eVal(1)
		.effect(AttachEnergyMe, Me).targetDeck().targetCardId(LIGHTNING_ENERGY).maxSelectCoinHeadCount()
		.effectShuffle()
		.attack(2248, u8"ピカボルト", u8"", "50", { Lightning, Lightning, Colorless })
		.textEn(u8"Pika Bolt", u8"");

	CreateCard(1699, u8"ピカチュウ", Pokemon, 1699)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 2)
		.weakness(Fighting)
		.attack(2249, u8"なんごくきぶん", u8"このポケモンを【ねむり】にする。自分の手札が6枚になるように、山札を引く。", "", { Colorless, Colorless })
		.textEn(u8"Tropical Vibes", u8"This Pokémon is now Asleep. Draw cards until you have 6 cards in your hand.")
		.postEffect(Sleep, Me)
		.effectDrawUntil(6);

	CreateCard(1700, u8"ピカチュウ", Pokemon, 1700)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2250, u8"こうそくいどう", u8"コインを1回投げオモテなら、次の相手の番、このポケモンはワザのダメージや効果を受けない。", "10", { Colorless })
		.textEn(u8"Agility", u8"Flip a coin. If heads, during your opponent’s next turn, prevent all damage from and effects of attacks done to this Pokémon.")
		.setPostEffect()
		.effectBreakIfCoinTail()
		.effectMe(NoDamageAndEffectAttackNextEnemyTurn);

	CreateCard(1701, u8"ピカチュウ", Pokemon, 1701)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2251, u8"よるのさんぽ", u8"自分の山札を1枚引く。", "", { Colorless })
		.textEn(u8"Nightime Stroll", u8"Draw a card.")
		.setPostEffect()
		.effectDraw(1)
		.attack(2252, u8"バチバチ", u8"", "20", { Lightning, Colorless })
		.textEn(u8"Static Shock", u8"");

	CreateCard(1702, u8"ピカチュウ", Pokemon, 1702)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 60, 1)
		.weakness(Fighting)
		.attack(2253, u8"かぜにあたる", u8"このポケモンの特殊状態を、すべて回復する。", "", { Colorless })
		.textEn(u8"Get Some Air", u8"This Pokémon recovers from all Special Conditions.")
		.postEffect(RecoverSpecialCondition, Me)
		.attack(2254, u8"けとばす", u8"", "20", { Colorless, Colorless })
		.textEn(u8"Smash Kick", u8"");

	CreateCard(1703, u8"ピカチュウ", Pokemon, 1703)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2255, u8"じゃれつく", u8"コインを1回投げオモテなら、20ダメージ追加。", "10+", { Colorless })
		.textEn(u8"Play Rough", u8"Flip a coin. If heads, this attack does 20 more damage.")
		.setPreEffect()
		.effectBreakIfCoinTail()
		.preEffectAttackDamageChange(20);

	CreateCard(1704, u8"ピカチュウ", Pokemon, 1704)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2256, u8"ためこむ", u8"自分のトラッシュから基本エネルギーを2枚まで選び、相手に見せて、手札に加える。", "", { Colorless })
		.textEn(u8"Store Up", u8"Put up to 2 Basic Energy cards from your discard pile into your hand.")
		.setPostEffect()
		.effectTrashToHand(2).targetBasicEnergy();

	CreateCard(1705, u8"ピカチュウ", Pokemon, 1705)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 80, 3)
		.weakness(Fighting)
		.attack(2257, u8"とうしのいかずち", u8"相手のバトルポケモンが「ポケモンex」なら、80ダメージ追加。", "20+", { Lightning, Colorless, Colorless })
		.textEn(u8"Fighting Lightning", u8"If your opponent’s Active Pokémon is a Pokémon ex, this attack does 80 more damage.")
		.setPreEffect()
		.exist(AreaType::Active, Enemy).targetCondition(TargetType::Ex)
		.preEffectAttackDamageChange(80);

	CreateCard(1706, u8"ピカチュウ", Pokemon, 1706)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 80, 3)
		.weakness(Fighting)
		.attack(2258, u8"まんぞくスパーク", u8"", "100", { Lightning, Lightning, Colorless, Colorless })
		.textEn(u8"Satisfied Spark", u8"");

	CreateCard(1707, u8"ピカチュウ", Pokemon, 1707)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 2)
		.weakness(Fighting)
		.attack(2259, u8"かみなりおとし", u8"このポケモンについている【雷】エネルギーをすべてトラッシュし、相手のポケモン1匹に、90ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Lightning, Lightning, Lightning })
		.textEn(u8"Lightning Crash", u8"Discard all {L} Energy from this Pokémon, and this attack does 90 damage to 1 of your opponent’s Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Lightning)
		.postEffectDamagePokemon(90);

	CreateCard(1708, u8"ピカチュウ", Pokemon, 1708)
		.nameEn(u8"Pikachu")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2260, u8"アンガーボルト", u8"このポケモンにのっているダメカンの数×10ダメージ追加。", "10+", { Lightning })
		.textEn(u8"Angry Bolt", u8"This attack does 10 more damage for each damage counter on this Pokémon.")
		.preEffectMe(AttackDamageChangeDamageCounter).eVal(10);

	CreateCard(1709, u8"ピカチュウex", Pokemon, 1709)
		.nameEn(u8"Pikachu ex")
		.pokemon(Ex, Basic, Lightning, 190, 1)
		.weakness(Fighting)
		.attack(2261, u8"ピカピカパレード", u8"自分の山札から【たね】ポケモンを好きなだけ選び、ベンチに出す。そして山札を切る。", "", { Colorless })
		.textEn(u8"Pika-Pika Parade", u8"Search your deck for any number of Basic Pokémon and put them onto your Bench. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToBenchAndShuffle(99).targetBasicPokemon()
		.attack(2262, u8"10まんボルト", u8"このポケモンについているエネルギーを、すべてトラッシュする。", "200", { Lightning, Lightning, Colorless })
		.textEn(u8"Thunderbolt", u8"Discard all Energy from this Pokémon.")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe);

	CreateCard(1710, u8"ピカチュウex", Pokemon, 1710)
		.nameEn(u8"Pikachu ex")
		.pokemon(Ex, Basic, Lightning, 190, 1)
		.weakness(Fighting)
		.attack(2263, u8"ビリビリフィーバー", u8"自分の手札から基本エネルギーを好きなだけ選び、自分のポケモンに好きなようにつける。", "", { Lightning })
		.textEn(u8"Zip-Zap Frenzy", u8"You may attach any number of Basic Energy cards from your hand to your Pokémon in any way you like.")
		.setPostEffect()
		.effectSelectAttachBasicEnergyHand(99)
		.effectAttachFromEach().targetPokemon()
		.attack(2264, u8"かみなり", u8"このポケモンにも30ダメージ。", "200", { Lightning, Lightning, Colorless })
		.textEn(u8"Thunder", u8"This Pokémon also does 30 damage to itself.")
		.postEffectDamageMe(30);

	CreateCard(1711, u8"サンダー", Pokemon, 1711)
		.nameEn(u8"Zapdos")
		.pokemon(Normal, Basic, Lightning, 120, 1)
		.weakness(Fighting)
		.activateSkillOnceTurn(561, u8"はじけるはばたき", u8"自分の場に「ファイヤー」「フリーザー」がいるなら、自分の番に1回使える。自分の手札から「基本【雷】エネルギー」を1枚選び、このポケモンにつける。")
		.textEn(u8" Flash-Pop Flapping", u8"Once during your turn, if you have Moltres and Articuno in play, you may use this Ability. Attach a Basic {L} Energy card from your hand to this Pokémon.")
		.existPokemon().targetName(u8"ファイヤー")
		.existPokemon().targetName(u8"フリーザー")
		.effectHandAttachEnergyMe().targetCardId(LIGHTNING_ENERGY)
		.attack(2265, u8"らいごう", u8"このポケモンにも60ダメージ。", "210", { Lightning, Lightning, Lightning, Colorless })
		.textEn(u8"Thundering Lightning", u8"This Pokémon also does 60 damage to itself.")
		.postEffectDamageMe(60);

	CreateCard(1712, u8"ゼクロム", Pokemon, 1712)
		.nameEn(u8"Zekrom")
		.pokemon(Normal, Basic, Lightning, 130, 2)
		.weakness(Fighting)
		.attack(2266, u8"きりさく", u8"", "50", { Lightning, Colorless })
		.textEn(u8"Slash", u8"")
		.attack(2267, u8"ニトロサンダー", u8"このポケモンに【炎】エネルギーがついているなら、80ダメージ追加。", "80+", { Lightning, Colorless, Colorless })
		.textEn(u8"Nitro Thunder", u8"If this Pokémon has any {R} Energy attaches, this attack does 80 more damage.")
		.setPreEffect()
		.conditionAttachEnergyMe(Fire)
		.preEffectAttackDamageChange(80);

	CreateCard(1713, u8"ゼラオラ", Pokemon, 1713)
		.nameEn(u8"Zeraora")
		.pokemon(Normal, Basic, Lightning, 110, 1)
		.weakness(Fighting)
		.attack(2268, u8"クイックドロー", u8"自分の山札を1枚引く。", "20", { Colorless })
		.textEn(u8"Rapid Draw", u8"Draw a card.")
		.setPostEffect()
		.effectDraw(1)
		.attack(2269, u8"エレキバレット", u8"相手のベンチポケモン1匹にも、20ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "50", { Lightning, Colorless })
		.textEn(u8"Electrobullet", u8"This attack also does 20 damage to 1 of your opponent’s Benched Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffectDamageBench(20);

	CreateCard(1714, u8"エレズン", Pokemon, 1714)
		.nameEn(u8"Toxel")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2270, u8"ぶつかる", u8"", "10", { Colorless })
		.textEn(u8"Ram", u8"");

	CreateCard(1715, u8"ストリンダー", Pokemon, 1715)
		.nameEn(u8"Toxtricity")
		.pokemon(Normal, Stage1, Lightning, 130, 2)
		.evolvesFrom(u8"エレズン")
		.weakness(Fighting)
		.attack(2271, u8"マッハボルト", u8"", "80", { Lightning, Colorless })
		.textEn(u8"Mach Bolt", u8"");

	CreateCard(1716, u8"ストリンダー", Pokemon, 1716)
		.nameEn(u8"Toxtricity")
		.pokemon(Normal, Stage1, Lightning, 140, 2)
		.evolvesFrom(u8"エレズン")
		.weakness(Fighting)
		.attack(2272, u8"なぐる", u8"", "40", { Lightning })
		.textEn(u8"Light Punch", u8"")
		.attack(2273, u8"サンダーボルト", u8"次の自分の番、このポケモンはワザが使えない。", "150", { Lightning, Colorless, Colorless })
		.textEn(u8"Thunderous Bolt", u8"During your next turn, this Pokémon can’t use attacks.")
		.postEffectMe(CannotAttackNextTurn);

	// Select a Snack: the top 3 are taken off the deck, 1 goes to the hand and the
	// others to the discard pile -- Explorer's Guidance with 3 and 1.
	CreateCard(1717, u8"モルペコ", Pokemon, 1717)
		.nameEn(u8"Morpeko")
		.pokemon(Normal, Basic, Lightning, 70, 1)
		.weakness(Fighting)
		.attack(2274, u8"おやつをえらぶ", u8"自分の山札を上から3枚トラッシュし、その中からカードを1枚選び、相手に見せて、手札に加える。", "", { Colorless })
		.textEn(u8"Select a Snack", u8"Discard the top 3 cards of your deck and put 1 of them into your hand.")
		.setPostEffect()
		.existMyDeck()
		.effect(LookDeck, Me).eVal(3)
		.effect(ToHand, Me).targetLooking().singleSelect()
		.effect(ToTrash, Me).targetLooking()
		.attack(2275, u8"ビンタ", u8"", "30", { Lightning })
		.textEn(u8"Slap", u8"");

	CreateCard(1718, u8"ミライドン", Pokemon, 1718)
		.nameEn(u8"Miraidon")
		.pokemon(Normal, Basic, Lightning, 120, 1)
		.weakness(Fighting)
		.attack(2276, u8"マッハボルト", u8"", "20", { Lightning })
		.textEn(u8"Mach Bolt", u8"")
		.attack(2277, u8"イナズマドライブ", u8"このポケモンについている【雷】エネルギーを2個選び、トラッシュする。", "140", { Lightning, Lightning, Colorless })
		.textEn(u8"Electro Drift", u8"Discard 2 {L} Energy from this Pokémon.")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Lightning).selectEnergy(2);

	CreateCard(1719, u8"ミュウツー", Pokemon, 1719)
		.nameEn(u8"Mewtwo")
		.pokemon(Normal, Basic, Psychic, 130, 2)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2278, u8"ちからをあたえる", u8"自分のトラッシュから基本エネルギーを2枚まで選び、自分のポケモン1匹につける。", "", { Psychic })
		.textEn(u8"Empower", u8"Attach up to 2 Basic Energy cards from your discard pile to 1 of your Pokémon.")
		.setPostEffect()
		.existPokemon()
		.effectSelectAttachBasicEnergyTrash(2)
		.effect(AttachSelectedCard, Me).targetPokemon().singleSelect()
		.attack(2279, u8"サイコドライブ", u8"このポケモンについているエネルギーを1個選び、トラッシュする。", "120", { Psychic, Psychic, Colorless })
		.textEn(u8"Psydrive", u8"Discard an Energy from this Pokémon.")
		.postEffectTrashEnergyMe(1);

	CreateCard(1720, u8"ミュウツーex", Pokemon, 1720)
		.nameEn(u8"Mewtwo ex")
		.pokemon(Ex, Basic, Psychic, 230, 2)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2280, u8"フォトンバレット", u8"相手の「ポケモンex」全員に、それぞれ50ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Psychic, Psychic })
		.textEn(u8"Photon Bullets", u8"This attack does 50 damage to each of your opponent’s Pokémon ex. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffect(AttackDamage, Enemy).eVal(50).targetPokemon().targetCondition(TargetType::Ex)
		.attack(2281, u8"サイキックフォース", u8"次の自分の番、このポケモンはワザが使えない。", "230", { Psychic, Psychic, Psychic })
		.textEn(u8"Psychic Powers", u8"During your next turn, this Pokémon can’t use attacks.")
		.postEffectMe(CannotAttackNextTurn);

	CreateCard(1721, u8"ミュウ", Pokemon, 1721)
		.nameEn(u8"Mew")
		.pokemon(Normal, Basic, Psychic, 60, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2282, u8"サイコキネシス", u8"相手のバトルポケモンについているエネルギーの数×40ダメージ追加。", "10+", { Psychic, Psychic })
		.textEn(u8"Psychic", u8"This attack does 40 more damage for each Energy attached to your opponent’s Active Pokémon.")
		.preEffect(AttackDamageChangeEnergyCount, Enemy).eVal(40).targetActive();

	CreateCard(1722, u8"マリル", Pokemon, 1722)
		.nameEn(u8"Marill")
		.pokemon(Normal, Basic, Psychic, 70, 1)
		.weakness(Metal)
		.attack(2283, u8"たいあたり", u8"", "30", { Psychic, Colorless })
		.textEn(u8"Tackle", u8"");

	CreateCard(1723, u8"マリルリ", Pokemon, 1723)
		.nameEn(u8"Azumarill")
		.pokemon(Normal, Stage1, Psychic, 130, 2)
		.evolvesFrom(u8"マリル")
		.weakness(Metal)
		.attack(2284, u8"のしかかり", u8"コインを1回投げオモテなら、相手のバトルポケモンを【マヒ】にする。", "90", { Psychic, Psychic, Colorless })
		.textEn(u8"Body Slam", u8"Flip a coin. If heads, your opponent’s Active Pokémon is now paralyzed.")
		.postEffectParalyzeIfCoinHead();

	CreateCard(1724, u8"エーフィ", Pokemon, 1724)
		.nameEn(u8"Espeon")
		.pokemon(Normal, Stage1, Psychic, 110, 1)
		.evolvesFrom(u8"イーブイ")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2285, u8"ミラクルシャイン", u8"相手の進化しているポケモン全員の上から、それぞれ「進化カード」を1枚ずつはがして退化させる。はがしたカードは、相手の手札にもどす。", "", { Psychic, Colorless })
		.textEn(u8"Miraculous Shine", u8"Devolve each of your opponent’s evolved Pokémon by putting the highest Stage Evolution card on it into your opponent’s hand.")
		.setPostEffect()
		.existPokemon(Enemy).targetCondition(TargetType::Evolved)
		.postEffect(Devolve, Enemy).eVal((int)AreaType::Hand).targetPokemon()
		.attack(2286, u8"ちょうねんりき", u8"", "90", { Psychic, Colorless, Colorless })
		.textEn(u8"Super Psy Bolt", u8"");

	CreateCard(1725, u8"エーフィex", Pokemon, 1725)
		.nameEn(u8"Espeon ex")
		.pokemon(Ex, Stage1, Psychic, 260, 1)
		.evolvesFrom(u8"イーブイ")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2287, u8"サンシャインビート", u8"自分の場のポケモンの数×30ダメージ。", "30×", { Psychic, Colorless })
		.textEn(u8"Solar Beatdown", u8"This attack does 30 damage for each of your Pokémon in play.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(30).targetPokemon();

	// Colorful Harmony: AttackDamageChangeTypeCount ORs the type of every target and
	// counts the bits, so aimed at your attached Basic Energy it counts their types.
	CreateCard(1726, u8"ニンフィアex", Pokemon, 1726)
		.nameEn(u8"Sylveon ex")
		.pokemon(Ex, Stage1, Psychic, 270, 2)
		.evolvesFrom(u8"イーブイ")
		.weakness(Metal)
		.attack(2288, u8"カラフルハーモニー", u8"自分のポケモン全員についている基本エネルギーのタイプの数×50ダメージ。", "50×", { Psychic, Colorless, Colorless })
		.textEn(u8"Colorful Harmony", u8"This attack does 50 damage for each type of Basic Energy attached to all of your Pokémon.")
		.preEffect(AttackDamageChangeTypeCount, Me).eVal(50).targetAttachedEnergy().targetBasicEnergy();

	CreateCard(1727, u8"アンノーン", Pokemon, 1727)
		.nameEn(u8"Unown")
		.pokemon(Normal, Basic, Psychic, 80, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2289, u8"ミステリーシグナル", u8"このワザのダメージで、相手のポケモンがきぜつしたなら、サイドを1枚多くとる。", "40", { Psychic, Psychic })
		.textEn(u8"Mysterious Signal", u8"If your opponent’s Pokémon is Knocked Out by damage from this attack, take 1 more Prize card.")
		.prizePlus1();

	CreateCard(1728, u8"フワンテ", Pokemon, 1728)
		.nameEn(u8"Drifloon")
		.pokemon(Normal, Basic, Psychic, 70, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2290, u8"まいあがる", u8"のぞむなら、このポケモンと、ついているすべてのカードを、山札にもどして切る。", "20", { Psychic })
		.textEn(u8"Float Up", u8"You may shuffle this Pokémon and all attached cards into your deck.")
		.postEffectSelectActivate()
		.effectMe(ToDeckWithAttach)
		.effectShuffle();

	CreateCard(1729, u8"クレセリア", Pokemon, 1729)
		.nameEn(u8"Cresselia")
		.pokemon(Normal, Basic, Psychic, 120, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2291, u8"オーロラゲイン", u8"このポケモンのHPを「30」回復する。", "30", { Psychic, Colorless })
		.textEn(u8"Aurora Gain", u8"Heal 30 damage from this Pokémon.")
		.postEffectMe(Heal).eVal(30)
		.attack(2292, u8"ルナブラスト", u8"", "100", { Psychic, Colorless, Colorless })
		.textEn(u8"Lunar Blast", u8"");

	CreateCard(1730, u8"シャンデラ", Pokemon, 1730)
		.nameEn(u8"Chandelure")
		.pokemon(Normal, Stage2, Psychic, 140, 2)
		.evolvesFrom(u8"ランプラー")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2293, u8"あやしいともしび", u8"相手のバトルポケモンを【やけど】と【こんらん】にする。", "130", { Psychic, Psychic })
		.textEn(u8"Eerie Glow", u8"Your opponent’s Active Pokémon is now Burned and Confused.")
		.postEffect(Burn, Enemy)
		.effect(Confuse, Enemy);

	CreateCard(1731, u8"ゼルネアス", Pokemon, 1731)
		.nameEn(u8"Xerneas")
		.pokemon(Normal, Basic, Psychic, 120, 2)
		.weakness(Metal)
		.attack(2294, u8"ジオナビゲート", u8"自分の山札からスタジアムを2枚まで選び、相手に見せて、手札に加える。そして山札を切る。", "", { Colorless })
		.textEn(u8"Geonavigation", u8"Search your deck for up to 2 Stadium card, reveal them, and put them into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.effectDeckToHandAndShuffle(2).targetStadiumCard()
		.attack(2295, u8"オーロラホーン", u8"", "100", { Psychic, Psychic, Colorless })
		.textEn(u8"Aurora Horns", u8"");

	CreateCard(1732, u8"キュワワー", Pokemon, 1732)
		.nameEn(u8"Comfey")
		.pokemon(Normal, Basic, Psychic, 70, 1)
		.weakness(Metal)
		.attack(2296, u8"やすらぎアロマ", u8"自分のベンチポケモン1匹のHPを「80」回復する。", "", { Colorless })
		.textEn(u8"Comforting Aroma", u8"Heal 80 damage from 1 of your Benched Pokémon.")
		.setPostEffect()
		.existAnyTargetAfterEffect()
		.effect(Heal, Me).eVal(80).targetBench().targetDamaged().singleSelect()
		.attack(2297, u8"マジカルショット", u8"", "30", { Psychic })
		.textEn(u8"Magical Shot", u8"");

	CreateCard(1733, u8"コスモッグ", Pokemon, 1733)
		.nameEn(u8"Cosmog")
		.pokemon(Normal, Basic, Psychic, 60, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2298, u8"はねる", u8"", "10", { Colorless })
		.textEn(u8"Splash", u8"");

	CreateCard(1734, u8"コスモウム", Pokemon, 1734)
		.nameEn(u8"Cosmoem")
		.pokemon(Normal, Stage1, Psychic, 100, 3)
		.evolvesFrom(u8"コスモッグ")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2299, u8"かたまる", u8"次の相手の番、このポケモンが受けるワザのダメージは「-60」される。", "", { Colorless, Colorless })
		.textEn(u8"Stiffen", u8"During your opponent’s next turn, this Pokémon takes 60 less damage from attacks (after applying Weakness and Resistance).")
		.postEffectMe(TakeDamageChangeNextEnemyTurn).eVal(-60);

	CreateCard(1735, u8"ルナアーラ", Pokemon, 1735)
		.nameEn(u8"Lunala")
		.pokemon(Normal, Stage2, Psychic, 160, 2)
		.evolvesFrom(u8"コスモウム")
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2300, u8"ミッドナイトレイ", u8"自分のトラッシュにあるエネルギーの枚数×20ダメージ追加。", "20+", { Psychic })
		.textEn(u8"Midnight Ray", u8"This attack does 20 more damage for each Energy card in your discard pile.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(20).targetTrash().targetEnergyCard()
		.attack(2301, u8"ルナブラスト", u8"", "120", { Psychic, Colorless, Colorless })
		.textEn(u8"Lunar Blast", u8"");

	CreateCard(1736, u8"コレクレー", Pokemon, 1736)
		.nameEn(u8"Gimmighoul")
		.pokemon(Normal, Basic, Psychic, 60, 1)
		.weakness(Darkness)
		.resistance(Fighting)
		.attack(2302, u8"たくさんあるく", u8"コインを1回投げオモテなら、自分の山札から好きなカードを1枚選び、手札に加える。そして山札を切る。", "", { Colorless })
		.textEn(u8"Strolls So Much", u8"Flip a coin. If heads, search your deck for a card and put it into your hand. Then, shuffle your deck.")
		.setPostEffect()
		.existMyDeck()
		.effectBreakIfCoinTail()
		.effectDeckToHandReverseAndShuffle(1);

	CreateCard(1737, u8"グラードン", Pokemon, 1737)
		.nameEn(u8"Groudon")
		.pokemon(Normal, Basic, Fighting, 140, 4)
		.weakness(Grass)
		.attack(2303, u8"グラウンドブレイク", u8"自分のベンチポケモン全員にも、それぞれ20ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "250", { Fighting, Fighting, Fighting, Fighting, Fighting })
		.textEn(u8"Break Ground", u8"This attack also does 20 damage to each of your Benched Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffect(AttackDamage, Me).eVal(20).targetBench();

	CreateCard(1738, u8"ルカリオ", Pokemon, 1738)
		.nameEn(u8"Lucario")
		.pokemon(Normal, Stage1, Fighting, 120, 2)
		.evolvesFrom(u8"リオル")
		.weakness(Psychic)
		.attack(2304, u8"はどうだん", u8"相手のベンチポケモン1匹にも、60ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "100", { Fighting, Fighting, Colorless })
		.textEn(u8"Aura Sphere", u8"This attack also does 60 damage to 1 of your opponent’s Benched Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffectDamageBench(60);

	CreateCard(1739, u8"ルガルガン", Pokemon, 1739)
		.nameEn(u8"Lycanroc")
		.pokemon(Normal, Stage1, Fighting, 130, 2)
		.evolvesFrom(u8"イワンコ")
		.weakness(Grass)
		.attack(2305, u8"カウンター", u8"前の相手の番に、このポケモンが受けたワザのダメージと同じダメージ追加。", "10+", { Fighting })
		.textEn(u8"Counter", u8"If this Pokémon was damaged by an attack during your opponent’s last turn, this attack does that much more damage.")
		.preEffectMe(AttackDamageChangeTakeAttackDamagePreTurn)
		.attack(2306, u8"ロックスマッシュ", u8"", "80", { Fighting, Fighting })
		.textEn(u8"Boulder Crush", u8"");

	CreateCard(1740, u8"コライドン", Pokemon, 1740)
		.nameEn(u8"Koraidon")
		.pokemon(Normal, Basic, Fighting, 130, 2)
		.weakness(Psychic)
		.attack(2307, u8"けたぐり", u8"", "50", { Fighting, Fighting })
		.textEn(u8"Low Kick", u8"")
		.attack(2308, u8"アクセルブレイク", u8"このポケモンについている【闘】エネルギーを2個選び、トラッシュする。", "140", { Fighting, Fighting, Colorless })
		.textEn(u8"Collision Course", u8"Discard 2 {F} Energy from this Pokémon.")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Fighting).selectEnergy(2);

	CreateCard(1741, u8"ニドラン♀", Pokemon, 1741)
		.nameEn(u8"Nidoran♀")
		.pokemon(Normal, Basic, Darkness, 60, 1)
		.weakness(Fighting)
		.attack(2309, u8"なきごえ", u8"次の相手の番、このワザを受けたポケモンが使うワザのダメージは「-30」される。", "", { Colorless })
		.textEn(u8"Growl", u8"During your opponent’s next turn, attacks used by the Defending Pokémon do 30 less damage (before applying Weakness and Resistance).")
		.postEffect(DamageChangeNextTurn, Enemy).eVal(-30).targetActive()
		.attack(2310, u8"ずつき", u8"", "10", { Darkness })
		.textEn(u8"Headbutt", u8"");

	CreateCard(1742, u8"ニドリーナ", Pokemon, 1742)
		.nameEn(u8"Nidorina")
		.pokemon(Normal, Stage1, Darkness, 90, 2)
		.evolvesFrom(u8"ニドラン♀")
		.weakness(Fighting)
		.activateSkillOnceTurn(562, u8"ハッピーシェア", u8"自分の番に1回使える。自分のポケモン1匹のHPを「30」回復する。")
		.textEn(u8" Share Happiness", u8"Once during your turn, you may use this Ability. Heal 30 damage from 1 of your Pokémon.")
		.effectHealSingle(30)
		.attack(2311, u8"かみつく", u8"", "30", { Colorless, Colorless })
		.textEn(u8"Bite", u8"");

	CreateCard(1743, u8"アローラ ニャース", Pokemon, 1743)
		.nameEn(u8"Alolan Meowth")
		.pokemon(Normal, Basic, Darkness, 60, 1)
		.weakness(Grass)
		.attack(2312, u8"ネコにこばん", u8"自分の山札を1枚引く。", "10", {})
		.textEn(u8"Pay Day", u8"Draw a card.")
		.setPostEffect()
		.effectDraw(1);

	CreateCard(1744, u8"ゲンガーex", Pokemon, 1744)
		.nameEn(u8"Gengar ex")
		.pokemon(Ex, Stage2, Darkness, 280, 2)
		.evolvesFrom(u8"ゴースト")
		.weakness(Fighting)
		.abilityBattleField(563, u8"しのせんこく", u8"このポケモンが、相手のポケモンからワザのダメージを受けてきぜつしたとき、自分はコインを1回投げる。オモテなら、ワザを使ったポケモンをきぜつさせる。")
		.textEn(u8" Fainting Spell", u8"If this Pokémon is Knocket Out by damage from an attack from your opponent’s Pokémon, flip a coin. If heads, the Attacking Pokémon is Knocket Out.")
		.triggerMe(TriggerType::KoEnemyAttackDamage)
		.effectBreakIfCoinTail()
		.effectTriggerObject(Ko)
		.attack(2313, u8"カオスペイン", u8"相手のポケモン1匹に、ダメカンを13個のせる。", "", { Darkness, Darkness })
		.textEn(u8"Chaotic Pain", u8"Place 13 damage counters on 1 of your opponent’s Pokémon.")
		.postEffect(DamageCounter, Enemy).eVal(13).targetPokemon().singleSelect();

	CreateCard(1745, u8"ブラッキー", Pokemon, 1745)
		.nameEn(u8"Umbreon")
		.pokemon(Normal, Stage1, Darkness, 110, 1)
		.evolvesFrom(u8"イーブイ")
		.weakness(Grass)
		.attack(2314, u8"かたきうち", u8"前の相手の番に、ワザのダメージで、自分のポケモンがきぜつしていたなら、100ダメージ追加。", "30+", { Darkness })
		.textEn(u8"Retaliate", u8"If any of your Pokémon were Knocked Out by damage from an attack during your opponent’s last turn, this attack does 100 more damage.")
		.setPreEffect()
		.condition(ConditionType::KoAttackDamagePreEnemyTurn)
		.preEffectAttackDamageChange(100)
		.attack(2315, u8"やみのキバ", u8"", "100", { Darkness, Colorless, Colorless })
		.textEn(u8"Darkness Fang", u8"");

	CreateCard(1746, u8"ブラッキーex", Pokemon, 1746)
		.nameEn(u8"Umbreon ex")
		.pokemon(Ex, Stage1, Darkness, 270, 2)
		.evolvesFrom(u8"イーブイ")
		.weakness(Grass)
		.attack(2316, u8"ルナティッククロー", u8"相手のバトルポケモンにダメカンがのっているなら、140ダメージ追加。", "100+", { Darkness, Colorless })
		.textEn(u8"Lunatic Claw", u8"If your opponent’s Active Pokémon already has any damage counters on it, this attack does 140 more damage.")
		.setPreEffect()
		.exist(AreaType::Active, Enemy).targetDamaged()
		.preEffectAttackDamageChange(140);

	CreateCard(1747, u8"ヤミカラス", Pokemon, 1747)
		.nameEn(u8"Murkrow")
		.pokemon(Normal, Basic, Darkness, 80, 1)
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2317, u8"ちょっとつかむ", u8"コインを1回投げオモテなら、次の相手の番、このワザを受けたポケモンは、にげられない。", "20", { Darkness })
		.textEn(u8"Clumsily Clutch", u8"Flip a coin. If heads, during your opponent’s next turn, the Defending Pokémon can’t retreat.")
		.setPostEffect()
		.effectBreakIfCoinTail()
		.effect(CannotRetreatNextTurn, Enemy).targetActive();

	CreateCard(1748, u8"ズルッグ", Pokemon, 1748)
		.nameEn(u8"Scraggy")
		.pokemon(Normal, Basic, Darkness, 80, 2)
		.weakness(Grass)
		.attack(2318, u8"けちをつける", u8"相手は相手自身の手札をすべて山札にもどして切る。その後、相手は山札を4枚引く。", "", { Darkness })
		.textEn(u8"Nitpick", u8"Your opponent shuffles their hand into their deck and draws 4 cards.")
		.setPostEffect()
		.canHandToDeckAndShuffle(Enemy)
		.effect(ToDeckReverseAndShuffle, Enemy).targetHand()
		.separator()
		.effect(Draw, Enemy).eVal(4)
		.attack(2319, u8"どつく", u8"", "30", { Darkness, Colorless })
		.textEn(u8"Corkscrew Punch", u8"");

	CreateCard(1749, u8"ゾロア", Pokemon, 1749)
		.nameEn(u8"Zorua")
		.pokemon(Normal, Basic, Darkness, 70, 1)
		.weakness(Grass)
		.attack(2320, u8"やみのキバ", u8"", "40", { Darkness, Darkness })
		.textEn(u8"Darkness Fang", u8"");

	CreateCard(1750, u8"ゾロアーク", Pokemon, 1750)
		.nameEn(u8"Zoroark")
		.pokemon(Normal, Stage1, Darkness, 120, 1)
		.evolvesFrom(u8"ゾロア")
		.weakness(Grass)
		.abilityBench(564, u8"よるのぬけみち", u8"このポケモンがベンチにいるかぎり、自分のバトルポケモンのにげるためのエネルギーは、2個ぶん少なくなる。")
		.textEn(u8" Nighttime Byway", u8"As long as this Pokémon is on your Bench, your Active Pokémon’s Retreat Cost is 2 less.")
		.effect(RetreatCostChange, Me).eVal(-2).targetActive()
		.attack(2321, u8"スラッシュクロー", u8"", "90", { Darkness, Darkness, Colorless })
		.textEn(u8"Slashing Claw", u8"");

	CreateCard(1751, u8"モノズ", Pokemon, 1751)
		.nameEn(u8"Deino")
		.pokemon(Normal, Basic, Darkness, 70, 2)
		.weakness(Grass)
		.attack(2322, u8"かじる", u8"", "10", { Darkness })
		.textEn(u8"Gnaw", u8"")
		.attack(2323, u8"ずつき", u8"", "20", { Darkness, Colorless })
		.textEn(u8"Headbutt", u8"");

	CreateCard(1752, u8"ジヘッド", Pokemon, 1752)
		.nameEn(u8"Zweilous")
		.pokemon(Normal, Stage1, Darkness, 100, 2)
		.evolvesFrom(u8"モノズ")
		.weakness(Grass)
		.attack(2324, u8"かみつく", u8"", "20", { Darkness })
		.textEn(u8"Bite", u8"")
		.attack(2325, u8"ぶちかます", u8"", "50", { Darkness, Colorless })
		.textEn(u8"Hammer In", u8"");

	CreateCard(1753, u8"サザンドラ", Pokemon, 1753)
		.nameEn(u8"Hydreigon")
		.pokemon(Normal, Stage2, Darkness, 170, 2)
		.evolvesFrom(u8"ジヘッド")
		.weakness(Grass)
		.attack(2326, u8"みつくびバイト", u8"コインを3回投げ、オモテの数ぶん、相手のバトルポケモンについているエネルギーを選び、トラッシュする。", "", { Darkness })
		.textEn(u8"Three-Headed Bite", u8"Flip 3 coins. For each heads, discard an Energy from your opponent’s Active Pokémon.")
		.postEffect(Coin, None).eVal(3)
		.postEffect(ToTrash, Enemy).targetAttachedEnergy().targetCondition(TargetType::AttachedActivePokemon).selectEnergyCoinHeadCount()
		.attack(2327, u8"しっこくのキバ", u8"", "140", { Darkness, Colorless })
		.textEn(u8"Pitch-Black Fangs", u8"");

	CreateCard(1754, u8"ガラル ニャース", Pokemon, 1754)
		.nameEn(u8"Galarian Meowth")
		.pokemon(Normal, Basic, Metal, 70, 1)
		.weakness(Fire)
		.resistance(Grass)
		.attack(2328, u8"ネコにこばん", u8"自分の山札を1枚引く。", "10", { Colorless })
		.textEn(u8"Pay Day", u8"Draw a card.")
		.setPostEffect()
		.effectDraw(1)
		.attack(2329, u8"おたからラッシュ", u8"自分の手札の枚数×10ダメージ。", "10×", { Metal })
		.textEn(u8"Treasure Rush", u8"This attack does 10 damage for each card in your hand.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(10).targetHand();

	CreateCard(1755, u8"ジラーチex", Pokemon, 1755)
		.nameEn(u8"Jirachi ex")
		.pokemon(Ex, Basic, Metal, 160, 1)
		.weakness(Fire)
		.resistance(Grass)
		.attack(2330, u8"ねがいをかなえる", u8"自分の手札が7枚になるように、山札を引く。", "", { Colorless })
		.textEn(u8"Wish Granter", u8"Draw cards until you have 7 cards in your hand.")
		.setPostEffect()
		.effectDrawUntil(7)
		.attack(2331, u8"スピードスター", u8"このワザのダメージは、弱点・抵抗力と、相手のバトルポケモンにかかっている効果を計算しない。", "150", { Colorless, Colorless, Colorless })
		.textEn(u8"Swift", u8"This attack’s damage isn’t affected by Weakness or Resistance, or by any effects on your opponent’s Active Pokémon.")
		.noTargetEffectAndWeakness();

	CreateCard(1756, u8"ディアルガ", Pokemon, 1756)
		.nameEn(u8"Dialga")
		.pokemon(Normal, Basic, Metal, 130, 2)
		.weakness(Fire)
		.resistance(Grass)
		.attack(2332, u8"リバースクロック", u8"自分のトラッシュからポケモンと基本エネルギーを合計3枚まで選び、相手に見せて、山札にもどして切る。", "", { Colorless })
		.textEn(u8"Reversed Clock", u8"Shuffle up to 3 in any combination of Pokémon and Basic Energy cards from your discard pile into your deck.")
		.setPostEffect()
		.effectTrashToDeckAndShuffle(3).targetCondition(TargetType::PokemonOrBasicEnergy)
		.attack(2333, u8"ヘビーインパクト", u8"", "110", { Metal, Metal, Colorless })
		.textEn(u8"Heavy Impact", u8"");

	CreateCard(1757, u8"ナットレイ", Pokemon, 1757)
		.nameEn(u8"Ferrothorn")
		.pokemon(Normal, Stage1, Metal, 130, 3)
		.evolvesFrom(u8"テッシード")
		.weakness(Fire)
		.resistance(Grass)
		.attack(2334, u8"トゲでさす", u8"", "50", { Colorless, Colorless })
		.textEn(u8"Spike Sting", u8"")
		.attack(2335, u8"ドッカンニードル", u8"相手のポケモン全員に、それぞれ50ダメージ。このポケモンにも130ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Metal, Metal })
		.textEn(u8"Kaboom Needles", u8"This attack does 50 damage to each of your opponent’s Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.) This Pokémon also does 130 damage to itself.")
		.postEffect(AttackDamage, Enemy).eVal(50).targetPokemon()
		.postEffectDamageMe(130);

	CreateCard(1758, u8"ソルガレオ", Pokemon, 1758)
		.nameEn(u8"Solgaleo")
		.pokemon(Normal, Stage2, Metal, 170, 2)
		.evolvesFrom(u8"コスモウム")
		.weakness(Fire)
		.resistance(Grass)
		.activateSkillOnceTurnBench(565, u8"サンライズ", u8"このポケモンがベンチにいるなら、自分の番に1回使える。自分の山札から「基本【鋼】エネルギー」を2枚まで選び、このポケモンにつける。そして山札を切る。")
		.textEn(u8" Sunrise", u8"Once during your turn, if this Pokémon is on your Bench, you may use this Ability. Search your deck for up to 2 Basic {M} Energy cards and attach them to this Pokémon. Then, shuffle your deck.")
		.effectDeckAttachEnergyMeAndShuffle(2).targetCardId(METAL_ENERGY)
		.attack(2336, u8"メテオドライブ", u8"このポケモンについているエネルギーを、すべてトラッシュする。", "220", { Metal, Metal, Colorless, Colorless })
		.textEn(u8"Sunsteel Strike", u8"Discard all Energy from this Pokémon.")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe);

	CreateCard(1759, u8"ザシアン", Pokemon, 1759)
		.nameEn(u8"Zacian")
		.pokemon(Normal, Basic, Metal, 130, 2)
		.weakness(Fire)
		.resistance(Grass)
		.attack(2337, u8"ハードブレード", u8"このポケモンに「ポケモンのどうぐ」がついているなら、40ダメージ追加。", "20+", { Metal })
		.textEn(u8"Hardened Blade", u8"If this Pokémon has a Pokémon Tool attached, this attack does 40 more damage.")
		.setPreEffect()
		.exist(AreaType::Me).targetCondition(TargetType::IsAttachedTool)
		.preEffectAttackDamageChange(40)
		.attack(2338, u8"スラッシュダウン", u8"次の自分の番、このポケモンは「スラッシュダウン」が使えない。", "120", { Metal, Metal, Colorless })
		.textEn(u8"Slashing Strike", u8"During your next turn, this Pokémon can’t use Slashing Strike")
		.postEffectMe(CannotUseThisAttackNextTurn);

	CreateCard(1760, u8"ザマゼンタ", Pokemon, 1760)
		.nameEn(u8"Zamazenta")
		.pokemon(Normal, Basic, Metal, 130, 2)
		.weakness(Fire)
		.resistance(Grass)
		.attack(2339, u8"はじきおとす", u8"ダメージを与える前に、相手のバトルポケモンについている「ポケモンのどうぐ」をトラッシュする。", "20", { Metal })
		.textEn(u8"Fend Off", u8"Before doing damage, discard all Pokémon Tools from your opponent’s Active Pokémon.")
		.preEffect(ToTrash, Enemy).targetAttachedTool().targetCondition(TargetType::AttachedActivePokemon)
		.attack(2340, u8"シールドプレス", u8"次の相手の番、このポケモンが受けるワザのダメージは「-50」される。", "100", { Metal, Metal, Colorless })
		.textEn(u8"Shield Press", u8"During your opponent’s next turn, this Pokémon takes 50 less damage from attacks (after applying Weakness and Resistance).")
		.postEffectMe(TakeDamageChangeNextEnemyTurn).eVal(-50);

	// Celebration takes Prize cards the way the pool's prize-taking effects do
	// (PrizeToHand); it can only be used with exactly 30 cards in hand.
	CreateCard(1761, u8"サーフゴー", Pokemon, 1761)
		.nameEn(u8"Gholdengo")
		.pokemon(Normal, Stage1, Metal, 130, 2)
		.evolvesFrom(u8"コレクレー")
		.weakness(Fire)
		.resistance(Grass)
		.attack(2341, u8"セレブレイト", u8"自分の手札が30枚なら、自分のサイドを2枚とる。その後、自分の手札をすべて山札にもどして切る。", "", { Metal })
		.textEn(u8"Celebration", u8"if you have exactly 30 cards in your hand, take 2 Prize cards. If you do, shuffle your hand into your deck.")
		.setPostEffect()
		.condition(ConditionType::CountTarget, 30).targetPlayer(Me).targetHand()
		.effect(PrizeToHand, Me).targetPrize().multiSelect(2)
		.effect(ToDeckReverseAndShuffle, Me).targetHand()
		.attack(2342, u8"トリプルスマッシュ", u8"コインを3回投げ、オモテの数×50ダメージ。", "50×", { Metal })
		.textEn(u8"Triple Smash", u8"Flip 3 coins. This attack does 50 damage for each heads.")
		.preEffect(AttackDamageChangeCoin, None).eVal(3, 50);

	CreateCard(1762, u8"ボーマンダex", Pokemon, 1762)
		.nameEn(u8"Salamence ex")
		.pokemon(Ex, Stage2, Dragon, 330, 2)
		.evolvesFrom(u8"コモルー")
		.attack(2343, u8"とどろくよびごえ", u8"自分のトラッシュから【竜】ポケモンを3枚まで選び、ベンチに出す。", "", { Colorless })
		.textEn(u8"Booming Call", u8"Put up to 3 {N} Pokémon from your discard pile onto your Bench.")
		.setPostEffect()
		.notFullBench()
		.effect(ToBench, Me).targetTrash().targetPokemonCard().targetEnergyType(Dragon).maxSelect(3)
		.attack(2344, u8"りゅうのはどう", u8"自分の山札を上から2枚トラッシュする。", "240", { Fire, Water })
		.textEn(u8"Dragon Pulse", u8"Discard the top 2 cards of your deck.")
		.postEffect(DeckToTrash, Me).eVal(2);

	CreateCard(1763, u8"ジャラコ", Pokemon, 1763)
		.nameEn(u8"Jangmo-o")
		.pokemon(Normal, Basic, Dragon, 70, 1)
		.attack(2345, u8"いやなおと", u8"次の自分の番、このワザを受けたポケモンが受けるワザのダメージは「+30」される。", "", { Colorless })
		.textEn(u8"Screech", u8"During your next turn, the Defending Pokémon takes 30 more damage from attacks (after applying Weakness and Resistance).")
		.postEffectActiveEnemy(TakeDamageChangeNextMyTurnEnemy).eVal(30)
		.attack(2346, u8"ドラゴンクロー", u8"", "40", { Lightning, Fighting })
		.textEn(u8"Dragon Claw", u8"");

	CreateCard(1764, u8"ジャランゴ", Pokemon, 1764)
		.nameEn(u8"Hakamo-o")
		.pokemon(Normal, Stage1, Dragon, 90, 2)
		.evolvesFrom(u8"ジャラコ")
		.attack(2347, u8"するどいキバ", u8"", "20", { Colorless })
		.textEn(u8"Sharp Fang", u8"")
		.attack(2348, u8"ドラゴンクロー", u8"", "70", { Lightning, Fighting })
		.textEn(u8"Dragon Claw", u8"");

	CreateCard(1765, u8"ジャラランガ", Pokemon, 1765)
		.nameEn(u8"Kommo-o")
		.pokemon(Normal, Stage2, Dragon, 180, 2)
		.evolvesFrom(u8"ジャランゴ")
		.attack(2349, u8"ブレイジングアッパー", u8"", "250", { Lightning, Fighting, Colorless })
		.textEn(u8"Blazing Uppercut", u8"");

	CreateCard(1766, u8"ニャース", Pokemon, 1766)
		.nameEn(u8"Meowth")
		.pokemon(Normal, Basic, Colorless, 60, 1)
		.weakness(Fighting)
		.attack(2350, u8"ネコにこばん", u8"自分の山札を1枚引く。", "30", { Colorless, Colorless })
		.textEn(u8"Pay Day", u8"Draw a card.")
		.setPostEffect()
		.effectDraw(1);

	CreateCard(1767, u8"ガルーラ", Pokemon, 1767)
		.nameEn(u8"Kangaskhan")
		.pokemon(Normal, Basic, Colorless, 130, 2)
		.weakness(Fighting)
		.attack(2351, u8"いかり", u8"このポケモンにのっているダメカンの数×10ダメージ追加。", "20+", { Colorless, Colorless })
		.textEn(u8"Rage", u8"This attack does 10 more damage for each damage counter on this Pokémon.")
		.preEffectMe(AttackDamageChangeDamageCounter).eVal(10)
		.attack(2352, u8"メガトンパンチ", u8"", "100", { Colorless, Colorless, Colorless })
		.textEn(u8"Mega Punch", u8"");

	CreateCard(1768, u8"イーブイ", Pokemon, 1768)
		.nameEn(u8"Eevee")
		.pokemon(Normal, Basic, Colorless, 70, 1)
		.weakness(Fighting)
		.attack(2353, u8"くわえてかくす", u8"相手の手札を見て、その中からグッズを1枚選び、相手の山札の下にもどす。", "", { Colorless })
		.textEn(u8"Fetch and Hide", u8"Your opponent reveals their hand, and you put an Item card you find there on the bottom of your opponent’s deck.")
		.setPostEffect()
		.exist(AreaType::Hand, Enemy)
		.effect(ToLooking, Enemy).targetHand()
		.effect(ToDeckBottom, Enemy).targetLooking().targetItem().singleSelect().cannotNoSelect()
		.effect(ToHand, Enemy).targetLooking()
		.attack(2354, u8"たいあたり", u8"", "10", { Colorless })
		.textEn(u8"Tackle", u8"");

	CreateCard(1769, u8"イーブイ", Pokemon, 1769)
		.nameEn(u8"Eevee")
		.pokemon(Normal, Basic, Colorless, 70, 1)
		.weakness(Fighting)
		.attack(2355, u8"でんこうせっか", u8"コインを1回投げオモテなら、20ダメージ追加。", "20+", { Colorless, Colorless })
		.textEn(u8"Quick Attack", u8"Flip a coin. If heads, this attack does 20 more damage.")
		.setPreEffect()
		.effectBreakIfCoinTail()
		.preEffectAttackDamageChange(20);

	CreateCard(1770, u8"ププリン", Pokemon, 1770)
		.nameEn(u8"Igglybuff")
		.pokemon(Normal, Basic, Colorless, 30, 0)
		.weakness(Fighting)
		.attack(2356, u8"ぷにぷにサークル", u8"最大HPが「30」の自分のベンチポケモンの数×30ダメージ。", "30×", {})
		.textEn(u8"Bouncy Circle", u8"This attack does 30 damage for each of your Benched Pokémon that has a maximum HP of 30.")
		.preEffect(AttackDamageChangeTargetCount, Me).eVal(30).targetBench().targetCondition(TargetType::MaxHp, 30);

	CreateCard(1771, u8"ルギア", Pokemon, 1771)
		.nameEn(u8"Lugia")
		.pokemon(Normal, Basic, Colorless, 120, 2)
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2357, u8"エレメンタルブラスト", u8"このポケモンについている【炎】【水】【雷】エネルギーを1個ずつ選び、トラッシュする。", "250", { Fire, Water, Lightning })
		.textEn(u8"Elemental Blast", u8"Discard a {R} Energy, a {W} Energy, and a {L} Energy from this Pokémon.")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Fire).selectEnergy(1)
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Water).selectEnergy(1)
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe).targetEnergyTypeAttached(Lightning).selectEnergy(1);

	CreateCard(1772, u8"ヒスイ ゾロア", Pokemon, 1772)
		.nameEn(u8"Hisuian Zorua")
		.pokemon(Normal, Basic, Colorless, 60, 1)
		.weakness(Fighting)
		.attack(2358, u8"ひっかく", u8"", "20", { Colorless })
		.textEn(u8"Scratch", u8"");

	CreateCard(1773, u8"ヒスイ ゾロアーク", Pokemon, 1773)
		.nameEn(u8"Hisuian Zoroark")
		.pokemon(Normal, Stage1, Colorless, 120, 1)
		.evolvesFrom(u8"ヒスイ ゾロア")
		.weakness(Fighting)
		.attack(2359, u8"ひっかく", u8"", "30", { Colorless })
		.textEn(u8"Scratch", u8"")
		.attack(2360, u8"えんさのうず", u8"相手のバトルポケモンの残りHPが「50」になるように、ダメカンをのせる。", "", { Colorless, Colorless, Colorless })
		.textEn(u8"Swirling Resentment", u8"Place damage counters on your opponent’s Active Pokémon until its remaining HP is 50.")
		.postEffectActiveEnemy(DamageCounterHp).eVal(50);

	CreateCard(1774, u8"メテノ", Pokemon, 1774)
		.nameEn(u8"Minior")
		.pokemon(Normal, Basic, Colorless, 90, 2)
		.weakness(Lightning)
		.resistance(Fighting)
		.attack(2361, u8"メテオシュート", u8"このポケモンについているエネルギーをすべてトラッシュし、相手のポケモン1匹に、120ダメージ。［ベンチは弱点・抵抗力を計算しない。］", "", { Colorless, Colorless, Colorless })
		.textEn(u8"Shoot Meteors", u8"Discard all Energy from this Pokémon, and this attack does 120 damage to 1 of your opponent’s Pokémon. (Don’t apply Weakness and Resistance for Benched Pokémon.)")
		.postEffect(ToTrash, Me).targetAttachedEnergy().targetCondition(TargetType::AttachedMe)
		.postEffectDamagePokemon(120);

	// Out of set order: Mew ex belongs with Mewtwo/Mew at 1720-1721, but the
	// card ids are the wire format the agent and the saved replays are keyed
	// on, so a new card goes on the end rather than renumbering the block.
	//
	// Memory Helix is the reason this one arrived late. The "use it as this
	// attack" family (asMyBenchNPokemonAttack and friends) pays the *host*
	// attack's Energy and only borrows the effect; Memory Helix instead lets
	// Mew ex use a Benched attack at that attack's own cost, out of its own
	// Energy -- which is how canUsePreEvolutionAttack already works, so it is
	// enumerated alongside it in SetAttackEnergy (GameUtil.h).
	CreateCard(1775, u8"ミュウex", Pokemon, 1775)
		.nameEn(u8"Mew ex")
		.pokemon(Ex, Basic, Psychic, 160, 0)
		.weakness(Darkness)
		.resistance(Fighting)
		.benchPokemonAttacks()
		.abilityActive(566, u8"メモリーヘリックス", u8"このポケモンは、自分のベンチポケモンが持つワザを使える。［ワザを使うためのエネルギーは必要。］")
		.textEn(u8"Memory Helix", u8"This Pokémon can use the attacks of any of your Benched Pokémon. (You still need the necessary Energy to use each attack.)")
		.attack(2362, u8"テレポートバースト", u8"のぞむなら、このポケモンをベンチポケモンと入れ替える。", "30", { Psychic })
		.textEn(u8"Teleportation Burst", u8"You may switch this Pokémon with 1 of your Benched Pokémon.")
		.setPostEffect()
		.existMyBench()
		.postEffectSelectActivate()
		.effectSwitch(Me);
	// ==== END 2026 sets ====
}
