with source as (
    select * from raw_order_items
),

cleaned as (
    select
        item_id,
        order_id,
        product_id,
        quantity,
        unit_price,
        discount,
        line_total,
        round((unit_price - (unit_price * discount)), 2) as net_price
    from source
    where item_id is not null
)

select * from cleaned