import { Link } from 'react-router-dom'
import { useMemo, useState } from 'react'
import './Home.css'
import { concerts } from '../assets/concerts'

function parseConcertDate(dateStr) {
  const t = new Date(dateStr).getTime()
  return Number.isFinite(t) ? t : Number.POSITIVE_INFINITY
}

// matches SeatSelection lowest math (base * 0.65 * 0.90), rounded to nearest 5
function computeFromPrice(basePrice) {
  const raw = (basePrice ?? 0) * 0.65 * 0.9
  return Math.round(raw / 5) * 5
}

const SORTS = {
  FEATURED: 'featured',
  PRICE_ASC: 'price_asc',
  PRICE_DESC: 'price_desc',
  DATE_ASC: 'date_asc',
  NAME_ASC: 'name_asc',
}

function Home() {
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState(SORTS.FEATURED)
  const [hoverInfoId, setHoverInfoId] = useState(null)

  const displayedConcerts = useMemo(() => {
    const filtered = concerts.filter(concert =>
      concert.name.toLowerCase().includes(search.toLowerCase())
    )

    const sorted = [...filtered]

    switch (sortBy) {
      case SORTS.PRICE_ASC:
        sorted.sort((a, b) => (a.price ?? 0) - (b.price ?? 0))
        break
      case SORTS.PRICE_DESC:
        sorted.sort((a, b) => (b.price ?? 0) - (a.price ?? 0))
        break
      case SORTS.DATE_ASC:
        sorted.sort((a, b) => parseConcertDate(a.date) - parseConcertDate(b.date))
        break
      case SORTS.NAME_ASC:
        sorted.sort((a, b) => (a.name || '').localeCompare(b.name || ''))
        break
      case SORTS.FEATURED:
      default:
        break
    }

    return sorted
  }, [search, sortBy])

  return (
    <div className="home-container">
      <header className="home-header">
        <div className="home-header-top">
          <div className="logo">
            <span className="logo-icon">🦋</span>
            <span className="logo-text">Ticket Monarch</span>
          </div>

          <div className="home-controls">
            <div className="search-bar">
              <span className="search-icon">🔍</span>
              <input
                type="text"
                placeholder="Search artists..."
                className="search-input"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>

            <div className="sort-wrap">
              <label className="sort-label" htmlFor="sortBy">Sort: </label>
              <select
                id="sortBy"
                className="sort-select"
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
              >
                <option value={SORTS.FEATURED}>Featured</option>
                <option value={SORTS.PRICE_ASC}>Price: Low → High</option>
                <option value={SORTS.PRICE_DESC}>Price: High → Low</option>
                <option value={SORTS.DATE_ASC}>Date: Soonest</option>
                <option value={SORTS.NAME_ASC}>Artist: A → Z</option>
              </select>
            </div>
          </div>
        </div>

        <div className="header-separator"></div>
      </header>

      <main className="home-main">
        <div className="concerts-list">
          {displayedConcerts.length === 0 && (<p>No artists found.</p>)}

          {displayedConcerts.map(concert => {
            const fromPrice = computeFromPrice(concert.price)
            const isOpen = hoverInfoId === concert.id

            return (
              <div key={concert.id} className="concert-card">
                <img
                  src={concert.image}
                  alt={concert.name}
                  className="concert-image"
                />

                <div className="concert-info">
                  <h2 className="concert-name">{concert.name}</h2>

                  <div className="concert-details">
                    <span className="concert-date">
                      {concert.date}

                      <span
                        className="info-wrap"
                        onMouseEnter={() => setHoverInfoId(concert.id)}
                        onMouseLeave={() => setHoverInfoId(null)}
                      >
                        <button
                          type="button"
                          className="info-btn"
                          aria-label={`Starting price for ${concert.name}`}
                          onFocus={() => setHoverInfoId(concert.id)}
                          onBlur={() => setHoverInfoId(null)}
                        >
                          ℹ️
                        </button>

                        {isOpen && (
                          <span className="info-tooltip">
                            From: ${fromPrice}
                          </span>
                        )}
                      </span>
                    </span>

                    <p className="concert-event">{concert.eventName}</p>
                    <p className="concert-location">{concert.location}</p>
                  </div>
                </div>

                <Link to={`/seats/${concert.id}`} className="tickets-button">
                  Tickets →
                </Link>
              </div>
            )
          })}
        </div>
      </main>

      <footer className="site-disclaimer">
        <p>
          Disclaimer: This is a student project and is not affiliated with any artists,
          venues, or event organizers. All concert information is for demonstration purposes only.
        </p>
        <br />
      </footer>
    </div>
  )
}

export default Home