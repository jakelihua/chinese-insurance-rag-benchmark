"""Conservative automatic field proxy; NOT a semantic correctness oracle."""
import re

from condition_data import POLICIES, compact


def chinese_number(text):
    if text.isdigit():return int(text)
    digits=dict(zip('零〇一二两三四五六七八九',[0,0,1,2,2,3,4,5,6,7,8,9]))
    total,current=0,0
    for char in text:
        if char in digits:current=digits[char]
        elif char in '十百':total+=(current or 1)*{'十':10,'百':100}[char];current=0
        else:return None
    return total+current


def resolve_contrastive_and_negated_spans(text):
    """Resolve explicit contrastive corrections ('不是/并非/并不是 A，而是/实际是 B')
    and strip standalone negated duration phrases so negated numbers are not extracted as facts."""
    num_day_re = r'(?:第|满|达|已经|已)?\s*[0-9零〇一二两三四五六七八九十百]+\s*(?:天|日)'

    def _replace_contrastive(match):
        prefix = match.group(1)
        aff_span = match.group(2).strip()
        if re.match(r'^' + num_day_re, aff_span):
            return prefix + aff_span
        return aff_span

    resolved = re.sub(
        r'(?:并不是|并非|不是|非但不是)([^，。；！？,.;!?\d]{0,18}?)'
        + num_day_re
        + r'(?:[，、；,]?\s*)(?:而是|实际是|其实是|实际上是|实为|却是|而是在|实际是在)([^，。；！？,.;!?]{1,24})',
        _replace_contrastive,
        text,
    )
    resolved = re.sub(
        r'(?:并不是|并非|不是|不到|未满|不足|未达|未达到|尚不足|不超|未超过)'
        r'(?:保单生效|合同生效|生效|停用前序药物|停药|术后|手术后|离职后|解除劳动合同后|知道权利受侵害后|后|距离|已经|已|满|达|第){0,3}\s*'
        + num_day_re,
        '',
        resolved,
    )
    return resolved


def detect_parse_uncertainty(text):
    """Identify surface syntax patterns that exceed the deterministic day-slot regex grammar
    (e.g., non-day time units, calendar date spans, relative arithmetic against thresholds,
    negated/approximate bounds without affirmative day counts, or conflicting multi-day values).
    Allows callers to separate parser syntax limitations (PARSE_UNCERTAIN) from genuine
    business underspecification (DEGRADED_TO_CLARIFY)."""
    raw = compact(text)
    num = r'[0-9零〇一二两三四五六七八九十百]+'
    reasons = []
    if re.search(
        r'(?:生效|停药|停用前序药物|术后|手术后|离职|解除劳动合同|知道权利受侵害)(?:算起|后|起|满|已|已经)?(?:第)?'
        + num
        + r'(?:个星期|星期|周|个月|月|半年|年)',
        raw,
    ) or re.search(r'\d{1,2}月\d{1,2}[日号].{0,24}\d{1,2}月\d{1,2}[日号]', raw):
        reasons.append('unsupported_time_unit_or_calendar_date')
    if re.search(
        r'(?:(?:比|较)(?:等待期|观察期|洗脱期|预防期|期限)(?:少|短|多|长|超|超出|早|晚)'
        + num
        + r'(?:天|日))|(?:(?:等待期|观察期|洗脱期|预防期)(?:刚过|届满后|满后|前)'
        + num
        + r'(?:天|日))',
        raw,
    ):
        reasons.append('relative_arithmetic_duration')
    if re.search(
        r'(?:生效|停药|停用前序药物|术后|手术后|离职|解除劳动合同|知道权利受侵害)[^，。；！？]{0,18}?(?:第)?'
        + num
        + r'(?:天|日)\s*(?:或|还是|或者|以及|变成|改为|至|到)\s*(?:第)?'
        + num
        + r'(?:天|日)',
        raw,
    ):
        reasons.append('conflicting_multi_day_values')
    if re.search(
        r'(?:(?:保单生效|合同生效|生效|停用前序药物|停药|术后|手术后|离职|解除劳动合同)(?:后|算起|起|已经|已)?\s*(?:不到|未满|不足|未达|未达到|尚不足)\s*(?:第)?'
        + num
        + r'(?:天|日))|(?:'
        r'(?:不到|未满|不足|未达|未达到|尚不足)(?:保单生效|合同生效|生效|停用前序药物|停药|术后|手术后|离职后|解除劳动合同后|后|距离|已经|已|满|达|第){0,3}\s*'
        + num
        + r'(?:天|日))',
        raw,
    ):
        reasons.append('negated_or_bounded_range_without_exact_day')
    first = bool(re.search(r'(?:首次|第一次|头一次)(?:投保|购买|买)', raw))
    new = bool(re.search(r'重新投保|再次投保|非连续投保', raw))
    if first and new:
        reasons.append('contradictory_enrollment_polarity')
    return {
        'is_parse_uncertain': bool(reasons),
        'reasons': reasons,
    }


def extract_fields(text):
    text=resolve_contrastive_and_negated_spans(compact(text));fields={}
    matches=[sid for sid,p in POLICIES.items() if compact(p['title']) in text]
    fields['policy_id']=matches[0] if len(matches)==1 else None
    first=bool(re.search(r'(?:首次|第一次|头一次)(?:投保|购买|买)',text))
    new=bool(re.search(r'重新投保|再次投保|非连续投保',text))
    fields['enrollment']='first' if first and not new else 'new' if new and not first else None
    # Negation is resolved before testing the positive accident term.
    illness=bool(re.search(r'非意外|不是意外|并非意外|因疾病|因病|生病|患病|得病',text))
    accident='意外' in re.sub(r'非意外|不是意外|并非意外','',text)
    fields['cause']='illness' if illness and not accident else 'accident' if accident and not illness else None
    number=r'([0-9零〇一二两三四五六七八九十百]+)'
    day=re.findall(r'生效(?:算起|后|起)?(?:的)?(?:第)?'+number+r'(?:天|日)',text)
    if re.search(r'生效[^，。；！？]{0,18}?(?:第)?'+number+r'(?:天|日)\s*(?:或|还是|或者|以及|变成|改为)\s*(?:第)?'+number+r'(?:天|日)',text):
        day=[]
    fields['policy_day']=chinese_number(day[0]) if len(day)==1 else None
    wait=re.findall(r'(?:计划.{0,8}等待期(?:为|是)?|等待期为)'+number+r'(?:天|日)',text)
    if not wait:wait=re.findall(number+r'(?:天|日)(?:的)?等待期',text)
    fields['plan_wait_days']=chinese_number(wait[0]) if len(wait)==1 else None
    if re.search(r'未获|没有.{0,8}(?:批准|免等待期)|未.{0,5}批准',text):fields['approved_waiver']=False
    elif re.search(r'(?:已获|已经|已).{0,15}(?:批准|同意).{0,8}免等待期',text):fields['approved_waiver']=True
    else:fields['approved_waiver']=None
    return fields


def condition_check(text,frame,required):
    observed=extract_fields(text)
    failures=[field for field in required if observed.get(field)!=frame[field]]
    return {'observed_fields':observed,'mismatched_or_unresolved_fields':failures,
            'proxy_all_required_fields_preserved':not failures}
