Phase premise: propose exactly premise_count companies in `sector`; the run seed picks one, and the
rest of the run builds that company's Slack workspace. A premise has:
- company: its name, whose first word differs from the other premises' first words and from every name in used_names;
- niche, region, size and culture: what it does, where in `country` it is, how big it is (it employs `pool` people)
  and how it works;
- cast: the kinds of people who work there;
- staffing: how many people of each occupation in `occupations` it employs. The counts add up to exactly `pool`, each
  at most that occupation's count in `occupations`; code hires exactly these people.
Write in `language`.
